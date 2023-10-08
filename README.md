Eigenvalue estimation for *ph*-multigrid
========================================

For a given scenario created in the `Problem::setup_scenario()` function, this simple program will generate a `ph`-multigrid hierarchy using global coarsening. 

Laplace matrices are created for each level, for which eigenvalue estimates are computed using `PreconditionChebyshev`.

Estimates are output to `cout`, while triangulation and polynomial degrees are written to disk in `vtk` format for each level.

Setup
=====

You need a sufficiently recent version of the deal.II library. Configure this project as an in-source build as follows:

    cmake -DDEAL_II_DIR=/path/to/dealii .
    make
