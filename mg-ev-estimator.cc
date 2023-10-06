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

#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/signaling_nan.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <fstream>

using namespace dealii;


template <int dim>
class Solution : public Function<dim>
{
public:
  Solution()
    : Function<dim>()
  {
    Assert(dim > 1, ExcNotImplemented());
  }

  virtual double value(const Point<dim> &p, const unsigned int /*component*/) const override
  {
    const std::array<double, 2> polar = GeometricUtilities::Coordinates::to_spherical(Point<2>(p[0], p[1]));

    constexpr double alpha = 2./3.;
    return std::pow(polar[0], alpha) * std::sin(alpha * polar[1]);
  }
};



template <int dim, int spacedim = dim>
class Problem
{
public:
  Problem();

  void run();

private:
  void setup_scenario();
  void output_scenario();

  Triangulation<dim, spacedim> triangulation;

  hp::FECollection<dim>     fe_collection;
  DoFHandler<dim, spacedim> dof_handler;


  void setup_multigrid_matrices();
  void estimate_eigenvalues();
  void output_eigenvalues();

  using VectorType                 = Vector<double>;
  using LevelMatrixType            = SparseMatrix<double>;
  using SmootherPreconditionerType = PreconditionJacobi<LevelMatrixType>;
  // MatrixFree: using SmootherPreconditionerType = DiagonalMatrix<VectorType>;

  MGLevelObject<std::unique_ptr<LevelMatrixType>> mg_matrices;
};



template<int dim, int spacedim>
Problem<dim, spacedim>::Problem()
: dof_handler(triangulation)
{}



template<int dim, int spacedim>
void
Problem<dim, spacedim>::setup_scenario()
{
  {
    // set up L-shaped grid
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
  }

  // set up fe collection
  for (unsigned int degree = 1; degree <= 3; ++degree)
    fe_collection.push_back(FE_Q<dim>(degree));

  // hp-refine center part
  Assert(dim > 1, ExcMessage("Setup works only for dim > 1."));
  Assert(fe_collection.size() > 2, ExcMessage("We need at least two FEs."));
  for(const auto& cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    {
      // set all cells to second to last FE
      cell->set_active_fe_index(fe_collection.size() - 2);

      const auto& center = cell->center();
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

  triangulation.execute_coarsening_and_refinement();
  dof_handler.distribute_dofs(fe_collection);
}



template<int dim, int spacedim>
void
Problem<dim, spacedim>::output_scenario()
{
  Vector<float> fe_degrees(triangulation.n_active_cells());
  for(const auto& cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    fe_degrees[cell->global_active_cell_index()] = cell->get_fe().degree;

  DataOut<dim, spacedim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(fe_degrees, "fe_degrees");
  data_out.build_patches();

  std::ofstream output("scenario_" + Utilities::int_to_string(spacedim) + "d.vtk");
  data_out.write_vtk(output);
}



template<int dim, int spacedim>
void
Problem<dim, spacedim>::setup_multigrid_matrices()
{
  Assert(false, ExcNotImplemented());

  // setup hierarchy of "cheap" poisson matrices
  // with corresponding constraint objects including Dirichlet boundary values
}



template<int dim, int spacedim>
void
Problem<dim, spacedim>::estimate_eigenvalues()
{
  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;

  // Initialize smoothers.
  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; level++)
    {
      smoother_data[level].preconditioner = std::make_shared<SmootherPreconditionerType>();

      // MatrixFree: mg_matrices[level]->compute_inverse_diagonal(smoother_data[level].preconditioner->get_vector());
      smoother_data[level].preconditioner->initialize(*mg_matrices[level]);

      smoother_data[level].smoothing_range     = 20.;
      smoother_data[level].degree              = 5;
      smoother_data[level].eig_cg_n_iterations = 20;
    }

  // Estimate eigenvalues on all levels, i.e., all operators.
  std::vector<double> min_eigenvalues(max_level + 1, numbers::signaling_nan<double>());
  std::vector<double> max_eigenvalues(max_level + 1, numbers::signaling_nan<double>());
  for (unsigned int level = min_level + 1; level <= max_level; level++)
    {
      SmootherType chebyshev;
      chebyshev.initialize(*mg_matrices[level], smoother_data[level]);

      VectorType vec;

      // MatrixFree: mg_matrices[level]->initialize_dof_vector(vec);
      vec.reinit(dof_handler.n_dofs());

      const auto evs = chebyshev.estimate_eigenvalues(vec);

      min_eigenvalues[level] = evs.min_eigenvalue_estimate;
      max_eigenvalues[level] = evs.max_eigenvalue_estimate;
    }
}



template<int dim, int spacedim>
void
Problem<dim, spacedim>::output_eigenvalues()
{
  Assert(false, ExcNotImplemented());
}



template<int dim, int spacedim>
void
Problem<dim, spacedim>::run()
{
  setup_scenario();
  output_scenario();

  setup_multigrid_matrices();
  estimate_eigenvalues();
  output_eigenvalues();
}



int main()
{
  Problem<2> problem_2d;
  problem_2d.run();

  Problem<3> problem_3d;
  problem_3d.run();
}
