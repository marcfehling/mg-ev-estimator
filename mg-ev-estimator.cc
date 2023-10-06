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

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/filtered_iterator.h>

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



template<int dim, int spacedim = dim>
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
Problem<dim, spacedim>::run()
{
  setup_scenario();

  // operator
  // mgtransfer
  // estimate ev
  // output ev

  output_scenario();
}



int main()
{
  Problem<2> problem_2d;
  problem_2d.run();

  Problem<3> problem_3d;
  problem_3d.run();
}
