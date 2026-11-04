/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>


#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h> //funcion requerida para el step 4

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <string> //Para cambiar el nombre del archivo

#include <deal.II/lac/affine_constraints.h>
 
#include <deal.II/grid/grid_refinement.h>
 
#include <deal.II/numerics/error_estimator.h>

#include <filesystem> // Para crear y manejar carpetas
namespace fs = std::filesystem;

using namespace dealii;


//----------------------------------------------------------------
//  VARIABLES GLOBALES
const int ref_2d = 3, ref_3d = 2;
const std::string base = "solution_ref", extension = ".vtk";
const float refinamiento = 0.3, coaserning = 0.05;
//----------------------------------------------------------------

std::string generar_nombre_archivo (const int cycle, const int dimension){

  std::string folder = "results_" + std::to_string(dimension) + "d";
  
  fs::create_directories(folder);

  return folder + "/" + base + std::to_string(cycle) + extension;
}


template <int dim>
class Step6
{
public:
  Step6();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation;

  const FE_Q<dim>    fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


template <int dim>
Step6<dim>::Step6()
  : fe(/* polynomial degree = */ 2)
  , dof_handler(triangulation)
{}


template <int dim>
void Step6<dim>::make_grid(){
  //Defino un condicional para adapatar la geometria dependiendo de si es 2D o 3D debido a que debo definir el rectangulo con puntos, los cuales debo definir sus coordenadas
  Triangulation<dim> rect1, rect2;

  if constexpr (dim == 2){
    GridGenerator::subdivided_hyper_rectangle(rect1, {2, 1}, Point<dim>(0, 0.5), Point<dim>(1, 1.0));
    GridGenerator::subdivided_hyper_rectangle(rect2, {1, 1}, Point<dim>(0.5, 0), Point<dim>(1, 0.5));
  }
  else if constexpr (dim == 3){
    GridGenerator::subdivided_hyper_rectangle(rect1, {2, 1, 2}, Point<dim>(0, 0.5, 0), Point<dim>(1.0, 1.0, 1.0));
    GridGenerator::subdivided_hyper_rectangle(rect2, {1, 1, 2}, Point<dim>(0.5, 0, 0), Point<dim>(1.0, 0.5, 1.0));  
  }else Assert(false, ExcNotImplemented());

  GridGenerator::merge_triangulations(rect1, rect2, triangulation); 
  const unsigned int nivel_ref = (dim == 2 ? ref_2d : ref_3d);
  triangulation.refine_global(nivel_ref);

  //La asignación del ID del limite, se hara o se mantendra de igual manera debido a que al solo extruir la figura 2d en 3d no cambia la condición para los limites o bordes
  

  for(auto &cell : triangulation.active_cell_iterators())
  {
    for(unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (cell->face(f)->at_boundary())
      {
        const Point<dim> center = cell->face(f)->center();
        
        //Se va a realizar una adaptación para que las caras de las celdas que queden sobre los bordes en el eje z, sean adiabaticas.
        //Condiciones de neumann 0
        bool cara_z = false;
        if constexpr (dim==3){
          if (std::abs(center(2) - 0.0) < 1e-8 || std::abs(center(2) - 1.0) < 1e-8)
            cara_z = true;
        }
        if (cara_z){
          //Se le asigna un ID diferente que no se va a usar, para que adopte las condiciones de neumann
          cell->face(f)->set_boundary_id(2);
        }else{
          if (std::abs(center(1) - 0.0) < 1e-8 && center(0) > 0.5)
          cell->face(f)->set_boundary_id(1); // Este será phi = 1
          else
          cell->face(f)->set_boundary_id(0); // El resto será phi = 0
        }
      }
    }
  }
}

template <int dim>
void Step6<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  
  
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  
  VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(0), Functions::ZeroFunction<dim>(), constraints);
  VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(1), Functions::ConstantFunction<dim>(1.0), constraints);

  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void Step6<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  const double coef_difusion = 0.005;
  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const Point<dim> q_point = fe_values.quadrature_point(q_index);
          Tensor<1,dim> velocity;
          velocity[0] = -q_point(1);
          velocity[1] = q_point(0);
          if constexpr (dim == 3) velocity[2] = 0; //no definido por el problema
          
          for (const unsigned int i : fe_values.dof_indices()){

            cell_rhs(i) += 0; //Vector f o generación           

            for (const unsigned int j : fe_values.dof_indices()){
              cell_matrix(i, j) +=
                ((coef_difusion*(fe_values.shape_grad(i, q_index) * 
                 fe_values.shape_grad(j, q_index))) + 
                 (velocity * fe_values.shape_grad(j, q_index) * fe_values.shape_value(i, q_index)))
                 * fe_values.JxW(q_index); 
            }        
          }
        }
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

template <int dim>
void Step6<dim>::solve()
{
  SolverControl            solver_control(3000, 1e-6 * system_rhs.l2_norm());
  SolverGMRES<Vector<double>> solver(solver_control);
  
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);


  std::cout << solver_control.last_step()
            << " GMRES iterations needed to obtain convergence." << std::endl;
}

template <int dim>
void Step6<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
 
  KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim - 1>(fe.degree + 1),
                                     {},
                                     solution,
                                     estimated_error_per_cell);
 
  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  refinamiento,
                                                  coaserning);
 
  triangulation.execute_coarsening_and_refinement();
}

template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  const std::string filename = generar_nombre_archivo(cycle,dim);
  std::ofstream     output(filename);
  data_out.write_vtk(output);
  std::cout << "Output written to " << filename << std::endl;
}

template <int dim>
void Step6<dim>::run()
{
  const unsigned int lim_cycle = (dim == 2) ? 15 : 4;

  for (unsigned int cycle = 0; cycle < lim_cycle; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;
  
    if (cycle == 0)
      {
        make_grid();
      }
    else
      refine_grid();
  
  
    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
  
    setup_system();
  
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
  
    assemble_system();
    solve();
    output_results(cycle);
  }
}

int main()
{

  try
    {
      {
        Step6<2> laplace_problem_2d;
        laplace_problem_2d.run();
      }
      /*{
        Step6<3> laplace_problem_3d;
        laplace_problem_3d.run();
      }*/
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}