// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/signaling_nan.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;


struct Parameters
{
  /**
   * Choose the type of finite element to be used.
   *
   * Choices: Qp | Qp-iso-Q1
   */
  const std::string fe_type = "Qp";

  /**
   * Creates a collection of finite elements with polynomial degrees
   * 1,...,max_degree.
   */
  const unsigned int max_degree = 3;

  /**
   * Choose a predefined discretization.
   *
   * Choices: L-domain | two-coarse-cells
   */
  const std::string scenario_type = "L-domain";

  /**
   * Decide whether to apply homogeneous Dirichlet boundary conditions to the
   * constraints on each level, and subsequently to each level matrix.
   */
  const bool apply_homogeneous_dirichlet_bc = true;

  /**
   * Specify which preconditioner to use for the smoother.
   *
   * Choices: Jacobi | SSOR | ASM
   */
  const std::string smoother_preconditioner_type = "Jacobi";

  /**
   * Write the discretization of each level into a separate vtk file.
   */
  const bool write_vtk = true;

  /**
   * Write each level matrix into a separate csv file.
   */
  const bool write_level_matrices = false;
};



namespace
{
  template <int dim, int spacedim>
  unsigned int
  get_max_active_fe_degree(const DoFHandler<dim, spacedim> &dof_handler)
  {
    unsigned int max = 0;

    for (auto &cell : dof_handler.active_cell_iterators() |
                        IteratorFilters::LocallyOwnedCell())
      max = std::max(max, cell->get_fe().degree);

    return Utilities::MPI::max(max, MPI_COMM_WORLD);
  };



  template <int dim, int spacedim>
  void
  write_vtk(const DoFHandler<dim, spacedim> &dof_handler,
            const std::string               &filename)
  {
    Vector<float> fe_degrees(dof_handler.get_triangulation().n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
      fe_degrees[cell->global_active_cell_index()] = cell->get_fe().degree;

    DataOut<dim, spacedim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(fe_degrees, "fe_degrees");
    data_out.build_patches();

    std::ofstream output(filename);
    data_out.write_vtk(output);
  };
} // namespace



/**
 * Adopted from
 * https://github.com/peterrum/dealii-asm/blob/d998b9b344a19c9d2890e087f953c2f93e6546ae/include/preconditioners.h#L145.
 */
template <typename Number, int dim, int spacedim>
class PreconditionASM
{
private:
  enum class WeightingType
  {
    none,
    left,
    right,
    symm
  };

public:
  PreconditionASM(const DoFHandler<dim, spacedim> &dof_handler)
    : dof_handler(dof_handler)
    , weighting_type(WeightingType::symm)
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern  &global_sparsity_pattern)
  {
    SparseMatrixTools::restrict_to_cells(global_sparse_matrix,
                                         global_sparsity_pattern,
                                         dof_handler,
                                         blocks);

    for (auto &block : blocks)
      if (block.m() > 0 && block.n() > 0)
        block.gauss_jordan();
  }

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0.0;
    src.update_ghost_values();

    Vector<double> vector_src, vector_dst, vector_weights;

    VectorType weights;

    if (weighting_type != WeightingType::none)
      {
        weights.reinit(src);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            vector_weights.reinit(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] = 1.0;

            cell->distribute_local_to_global(vector_weights, weights);
          }

        weights.compress(VectorOperation::add);
        for (auto &i : weights)
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) :
                                                        (1.0 / i);
        weights.update_ghost_values();
      }

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);
        if (weighting_type != WeightingType::none)
          vector_weights.reinit(dofs_per_cell);

        cell->get_dof_values(src, vector_src);
        if (weighting_type != WeightingType::none)
          cell->get_dof_values(weights, vector_weights);

        if (weighting_type == WeightingType::symm ||
            weighting_type == WeightingType::right)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_src[i] *= vector_weights[i];

        blocks[cell->active_cell_index()].vmult(vector_dst, vector_src);

        if (weighting_type == WeightingType::symm ||
            weighting_type == WeightingType::left)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_dst[i] *= vector_weights[i];

        cell->distribute_local_to_global(vector_dst, dst);
      }

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  }

private:
  const DoFHandler<dim, spacedim> &dof_handler;
  std::vector<FullMatrix<Number>>  blocks;

  const WeightingType weighting_type;
};



template <int dim, int spacedim = dim>
class Problem
{
public:
  Problem();

  void
  run();

private:
  void
  setup_scenario();

  void
  setup_mg_matrices();

  template <typename SmootherPreconditionerType>
  void
  estimate_eigenvalues();

  void
  output();

  Parameters prm;

  Triangulation<dim, spacedim> triangulation;
  DoFHandler<dim, spacedim>    dof_handler;

  hp::FECollection<dim, spacedim> fe_collection;
  hp::QCollection<dim>            quadrature_collection;

  using VectorType      = Vector<double>;
  using LevelMatrixType = SparseMatrix<double>;

  std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
                                           mg_triangulations;
  MGLevelObject<DoFHandler<dim, spacedim>> mg_dof_handlers;
  MGLevelObject<SparsityPattern>           mg_sparsity_patterns;
  MGLevelObject<LevelMatrixType>           mg_matrices;

  std::vector<double> min_eigenvalues;
  std::vector<double> max_eigenvalues;
};



template <int dim, int spacedim>
Problem<dim, spacedim>::Problem()
  : dof_handler(triangulation)
{}



template <int dim, int spacedim>
void
Problem<dim, spacedim>::setup_scenario()
{
  // set up collections
  for (unsigned int degree = 1; degree <= prm.max_degree; ++degree)
    if (prm.fe_type == "Qp")
      {
        fe_collection.push_back(FE_Q<dim, spacedim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
      }
    else if (prm.fe_type == "Qp-iso-Q1")
      {
        fe_collection.push_back(FE_Q_iso_Q1<dim, spacedim>(degree));
        quadrature_collection.push_back(QIterated<dim>(QGauss<1>(2), degree));
      }
    else
      AssertThrow(false, ExcNotImplemented());

  // set up discretization
  if (prm.scenario_type == "L-domain")
    {
      std::vector<unsigned int> repetitions(dim);
      Point<dim>                bottom_left, top_right;
      for (unsigned int d = 0; d < dim; ++d)
        if (d < 2)
          {
            repetitions[d] = 2;
            bottom_left[d] = -1.;
            top_right[d]   = 1.;
          }
        else
          {
            repetitions[d] = 1;
            bottom_left[d] = 0.;
            top_right[d]   = 1.;
          }

      std::vector<int> cells_to_remove(dim, 1);
      cells_to_remove[0] = -1;

      GridGenerator::subdivided_hyper_L(
        triangulation, repetitions, bottom_left, top_right, cells_to_remove);

      triangulation.refine_global(2);

      // hp-refine center part
      Assert(dim > 1, ExcMessage("Setup works only for dim > 1."));
      Assert(fe_collection.size() > 2, ExcMessage("We need at least two FEs."));
      for (const auto &cell : dof_handler.active_cell_iterators() |
                                IteratorFilters::LocallyOwnedCell())
        {
          // set all cells to second to last FE
          cell->set_active_fe_index(fe_collection.size() - 2);

          const auto &center = cell->center();
          if (std::abs(center[0]) < 0.5 && std::abs(center[1]) < 0.5)
            {
              if (center[0] < -0.25 || center[1] > 0.25)
                // outer layer gets p-refined
                cell->set_active_fe_index(fe_collection.size() - 1);
              else
                // inner layer gets h-refined
                cell->set_refine_flag();
            }
        }
    }
  else if (prm.scenario_type == "two-coarse-cells")
    {
      std::vector<unsigned int> repetitions(dim);
      Point<dim>                bottom_left, top_right;
      for (unsigned int d = 0; d < dim; ++d)
        if (d < 1)
          {
            repetitions[d] = 2;
            bottom_left[d] = 0.;
            top_right[d]   = 2.;
          }
        else
          {
            repetitions[d] = 1;
            bottom_left[d] = 0.;
            top_right[d]   = 1.;
          }

      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                bottom_left,
                                                top_right);

      // hp-refine
      Assert(fe_collection.size() > 2, ExcMessage("We need at least two FEs."));
      for (const auto &cell : dof_handler.active_cell_iterators() |
                                IteratorFilters::LocallyOwnedCell())
        {
          // set all cells to second to last FE
          cell->set_active_fe_index(fe_collection.size() - 2);

          if (std::abs(cell->center()[0]) < 1)
            // left cell gets p-refined
            cell->set_active_fe_index(fe_collection.size() - 1);
          else
            // right cell gets h-refined
            cell->set_refine_flag();
        }
    }
  else
    AssertThrow(false, ExcNotImplemented());

  triangulation.execute_coarsening_and_refinement();
  dof_handler.distribute_dofs(fe_collection);
}



template <int dim, int spacedim>
void
Problem<dim, spacedim>::setup_mg_matrices()
{
  const auto p_sequence = MGTransferGlobalCoarseningTools::
    PolynomialCoarseningSequenceType::decrease_by_one;

  std::map<unsigned int, unsigned int> fe_index_for_degree;
  for (unsigned int i = 0; i < fe_collection.size(); ++i)
    fe_index_for_degree[dof_handler.get_fe(i).degree] = i;

  mg_triangulations =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
      triangulation);

  const unsigned int n_h_levels = mg_triangulations.size() - 1;
  const unsigned int n_p_levels =
    MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
      get_max_active_fe_degree(dof_handler), p_sequence)
      .size();

  const unsigned int minlevel = 0;
  const unsigned int maxlevel = n_h_levels + n_p_levels - 1;

  mg_dof_handlers.resize(minlevel, maxlevel);
  mg_sparsity_patterns.resize(minlevel, maxlevel);
  mg_matrices.resize(minlevel, maxlevel);

  // Loop from max to min level and set up DoFHandler with coarser mesh...
  for (unsigned int l = 0; l < n_h_levels; ++l)
    {
      mg_dof_handlers[l].reinit(*mg_triangulations[l]);
      mg_dof_handlers[l].distribute_dofs(fe_collection);
    }

  // ... with lower polynomial degrees
  for (unsigned int i = 0, l = maxlevel; i < n_p_levels; ++i, --l)
    {
      mg_dof_handlers[l].reinit(triangulation);

      if (l == maxlevel) // finest level
        {
          auto cell_other = dof_handler.begin_active();
          for (const auto &cell : mg_dof_handlers[l].active_cell_iterators())
            {
              if (cell->is_locally_owned())
                cell->set_active_fe_index(cell_other->active_fe_index());
              cell_other++;
            }
        }
      else // coarse level
        {
          auto &dof_handler_fine   = mg_dof_handlers[l + 1];
          auto &dof_handler_coarse = mg_dof_handlers[l + 0];

          auto cell_other = dof_handler_fine.begin_active();
          for (const auto &cell : dof_handler_coarse.active_cell_iterators())
            {
              if (cell->is_locally_owned())
                {
                  const unsigned int next_degree =
                    MGTransferGlobalCoarseningTools::
                      create_next_polynomial_coarsening_degree(
                        cell_other->get_fe().degree, p_sequence);

                  cell->set_active_fe_index(fe_index_for_degree[next_degree]);
                }
              cell_other++;
            }
        }

      mg_dof_handlers[l].distribute_dofs(fe_collection);
    }

  MGLevelObject<AffineConstraints<typename LevelMatrixType::value_type>>
    mg_constraints(minlevel, maxlevel);
  for (unsigned int level = minlevel; level <= maxlevel; level++)
    {
      const auto &dof_handler      = mg_dof_handlers[level];
      auto       &constraints      = mg_constraints[level];
      auto       &sparsity_pattern = mg_sparsity_patterns[level];
      auto       &matrix           = mg_matrices[level];

      // Note: the following part needs to be adjusted for parallel applications

      // ... constraints (with homogenous Dirichlet BC -- optional)
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      if (prm.apply_homogeneous_dirichlet_bc == true)
        {
          VectorTools::interpolate_boundary_values(
            dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
        }
      constraints.close();

      // ... matrices
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      constraints,
                                      /*keep_constrained_dofs = */ true);
      sparsity_pattern.copy_from(dsp);

      matrix.reinit(sparsity_pattern);
      MatrixCreator::create_laplace_matrix(
        dof_handler,
        quadrature_collection,
        matrix,
        (const Function<spacedim, typename LevelMatrixType::value_type>
           *const)nullptr,
        constraints);
    }

  for (unsigned int level = 0; level <= maxlevel; ++level)
    {
      if (mg_dof_handlers[level].n_dofs() > 10000)
        continue;

      std::vector<Point<dim>> support(mg_dof_handlers[level].n_dofs());
      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                           mg_dof_handlers[level],
                                           support);
      std::vector<unsigned int> at_interface;
      for (unsigned int i = 0; i < support.size(); ++i)
        if (std::abs(support[i][0] - 1.) < 1e-12)
          at_interface.push_back(i);
      const unsigned int       m = mg_matrices[level].m();
      LAPACKFullMatrix<double> matrix(m, m);
      for (auto it = mg_matrices[level].begin(); it != mg_matrices[level].end();
           ++it)
        matrix(it->row(), it->column()) = it->value();
      for (unsigned int i = 0; i < m; ++i)
        for (unsigned int j = 0; j < i; ++j)
          if (std::abs(matrix(i, j) - matrix(j, i)) > 1e-12)
            std::cout << "Matrix (" << i << " " << j
                      << ") not symmetric: " << matrix(i, j) << " vs "
                      << matrix(j, i) << std::endl;

      LAPACKFullMatrix<double> diagonal(m, m);
      for (unsigned int i = 0; i < m; ++i)
        diagonal(i, i) = matrix(i, i);

      // Invert block of dofs at interface
      if (false && prm.scenario_type == "two-coarse-cells")
        for (unsigned int i : at_interface)
          for (unsigned int j : at_interface)
            diagonal(i, j) = matrix(i, j);

      std::vector<Vector<double>> eigenvectors(m, Vector<double>(m));
      matrix.compute_generalized_eigenvalues_symmetric(diagonal, eigenvectors);
      std::cout << "eigenvalues: ";
      std::cout << matrix.eigenvalue(0).real() << " "
                << matrix.eigenvalue(m - 3).real() << " "
                << matrix.eigenvalue(m - 2).real() << " "
                << matrix.eigenvalue(m - 1).real() << std::endl;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      DataOut<dim, spacedim> data_out;
      data_out.set_flags(flags);
      for (Vector<double> &vec : eigenvectors)
        mg_constraints[level].distribute(vec);

      data_out.attach_dof_handler(mg_dof_handlers[level]);
      data_out.add_data_vector(eigenvectors[0], "eig0");
      data_out.add_data_vector(eigenvectors[m - 3], "eigm3");
      data_out.add_data_vector(eigenvectors[m - 2], "eigm2");
      data_out.add_data_vector(eigenvectors[m - 1], "eigm1");
      data_out.build_patches(dim == 2 ? 60 : 40);

      std::ofstream output("eigenvalues-d" + std::to_string(dim) + "-l" +
                           std::to_string(level) + ".vtk");
      data_out.write_vtk(output);
    }
}



template <int dim, int spacedim>
template <typename SmootherPreconditionerType>
void
Problem<dim, spacedim>::estimate_eigenvalues()
{
  using SmootherType = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;

  // Initialize smoothers.
  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level,
                                                                     max_level);
  for (unsigned int level = min_level; level <= max_level; level++)
    {
      if constexpr (std::is_same_v<SmootherPreconditionerType,
                                   PreconditionASM<double, dim, spacedim>>)
        {
          smoother_data[level].preconditioner =
            std::make_shared<SmootherPreconditionerType>(
              mg_dof_handlers[level]);

          smoother_data[level].preconditioner->initialize(
            mg_matrices[level], mg_sparsity_patterns[level]);
        }
      else
        {
          smoother_data[level].preconditioner =
            std::make_shared<SmootherPreconditionerType>();

          smoother_data[level].preconditioner->initialize(mg_matrices[level]);
        }

      smoother_data[level].smoothing_range     = 20.;
      smoother_data[level].degree              = 5;
      smoother_data[level].eig_cg_n_iterations = 20;
    }

  // Estimate eigenvalues on all levels, i.e., all operators.
  min_eigenvalues.resize(max_level + 1, numbers::signaling_nan<double>());
  max_eigenvalues.resize(max_level + 1, numbers::signaling_nan<double>());
  for (unsigned int level = min_level; level <= max_level; level++)
    {
      SmootherType chebyshev;
      chebyshev.initialize(mg_matrices[level], smoother_data[level]);

      VectorType vec;
      vec.reinit(mg_dof_handlers[level].n_dofs());

      const auto evs = chebyshev.estimate_eigenvalues(vec);

      min_eigenvalues[level] = evs.min_eigenvalue_estimate;
      max_eigenvalues[level] =
        evs.min_eigenvalue_estimate < evs.max_eigenvalue_estimate ?
          evs.max_eigenvalue_estimate / 1.2 :
          evs.max_eigenvalue_estimate;
    }
}



template <int dim, int spacedim>
void
Problem<dim, spacedim>::output()
{
  const std::string description =
    Utilities::int_to_string(dim) + "d_" + prm.scenario_type + "_" +
    prm.fe_type + "_" + prm.smoother_preconditioner_type +
    (prm.apply_homogeneous_dirichlet_bc ? "_BC" : "");

  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  ConvergenceTable table;
  for (unsigned int level = min_level; level <= max_level; ++level)
    {
      table.add_value("mg", level);
      table.add_value("tria_levels",
                      mg_dof_handlers[level].get_triangulation().n_levels());
      table.add_value("max_degree",
                      get_max_active_fe_degree(mg_dof_handlers[level]));
      table.add_value("n_dofs", mg_dof_handlers[level].n_dofs());
      table.add_value("min_eigenvalue", min_eigenvalues[level]);
      table.add_value("max_eigenvalue", max_eigenvalues[level]);

      const std::string filestem =
        description + "_level-" + Utilities::int_to_string(level);

      if (prm.write_level_matrices == true)
        {
          std::ofstream output(filestem + ".csv");
          mg_matrices[level].print_formatted(output, 3, true, 0, " ", 1., ",");
        }

      if (prm.write_vtk == true)
        write_vtk(mg_dof_handlers[level], filestem + ".vtk");
    }

  std::cout << description << std::endl;
  table.write_text(std::cout);
}



template <int dim, int spacedim>
void
Problem<dim, spacedim>::run()
{
  setup_scenario();

  setup_mg_matrices();

  if (prm.smoother_preconditioner_type == "Jacobi")
    estimate_eigenvalues<PreconditionJacobi<LevelMatrixType>>();
  else if (prm.smoother_preconditioner_type == "SSOR")
    estimate_eigenvalues<PreconditionSSOR<LevelMatrixType>>();
  else if (prm.smoother_preconditioner_type == "ASM")
    estimate_eigenvalues<PreconditionASM<double, dim, spacedim>>();
  else
    AssertThrow(false, ExcNotImplemented());

  output();
}



int
main()
{
  Problem<2> problem_2d;
  problem_2d.run();

  Problem<3> problem_3d;
  problem_3d.run();
}
