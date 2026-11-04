#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;
using namespace dealii;

//  VARIABLES GLOBALES
const int ref_2d = 2, ref_3d = 3;
// CORRECCIÓN 1: extensión cambiada a .vtu para ser consistente con write_vtu
const std::string base      = "solution_ref";
const std::string extension = ".vtu";
const float refinamiento = 0.2, coaserning = 0.1;

std::string generar_nombre_archivo(const int         cycle,
                                   const int         dimension,
                                   const std::string &prefix = "")
{
  std::string folder = "results_" + std::to_string(dimension) + "d";
  fs::create_directories(folder);
  return folder + "/" + prefix + base + std::to_string(cycle) + extension;
}

template <int dim>
class AdvectionField : public TensorFunction<1, dim>
{
public:
  virtual Tensor<1, dim> value(const Point<dim> &p) const override;
};

template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
{
  Tensor<1, dim> value;
  value[0] = -p[1];
  value[1] =  p[0];
  if (dim == 3)
    value[2] = 0.0;
  return value;
}

class GradientEstimation
{
public:
  template <int dim>
  static void estimate(const FiniteElement<dim> &fe,
                       const DoFHandler<dim>    &dof,
                       const Vector<double>     &solution,
                       Vector<float>            &error_per_cell);

  DeclException2(ExcInvalidVectorLength,
                 int,
                 int,
                 << "Vector has length " << arg1 << ", but should have " << arg2);
  DeclException0(ExcInsufficientDirections);
};

template <int dim>
class Step9
{
public:
  Step9();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void refine_grid();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim>        triangulation;
  const FE_Q<dim>           fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;
  Vector<double>            solution;
  Vector<double>            system_rhs;
};

template <int dim>
Step9<dim>::Step9()
  : fe(3)  // Polinomios de grado 5
  , dof_handler(triangulation)
{}

template <int dim>
void Step9<dim>::make_grid()
{
  Triangulation<dim> rect1, rect2;

  if constexpr (dim == 2)
    {
      GridGenerator::subdivided_hyper_rectangle(rect1, {2, 1},
                                                Point<dim>(0, 0.5),
                                                Point<dim>(1, 1.0));
      GridGenerator::subdivided_hyper_rectangle(rect2, {1, 1},
                                                Point<dim>(0.5, 0),
                                                Point<dim>(1, 0.5));
    }
  else if constexpr (dim == 3)
    {
      GridGenerator::subdivided_hyper_rectangle(rect1, {2, 1, 2},
                                                Point<dim>(0, 0.5, 0),
                                                Point<dim>(1.0, 1.0, 1.0));
      GridGenerator::subdivided_hyper_rectangle(rect2, {1, 1, 2},
                                                Point<dim>(0.5, 0, 0),
                                                Point<dim>(1.0, 0.5, 1.0));
    }
  else
    Assert(false, ExcNotImplemented());

  GridGenerator::merge_triangulations(rect1, rect2, triangulation);
  const unsigned int nivel_ref = (dim == 2 ? ref_2d : ref_3d);
  triangulation.refine_global(nivel_ref);

  for (auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
          const Point<dim> center = cell->face(f)->center();
          bool             cara_z = false;
          if constexpr (dim == 3)
            if (std::abs(center(2) - 0.0) < 1e-8 ||
                std::abs(center(2) - 1.0) < 1e-8)
              cara_z = true;

          if (cara_z)
            cell->face(f)->set_boundary_id(2);
          else if (std::abs(center(1) - 0.0) < 1e-8 && center(0) > 0.5)
            cell->face(f)->set_boundary_id(1); // phi = 1
          else
            cell->face(f)->set_boundary_id(0); // phi = 0
        }
}

template <int dim>
void Step9<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
    dof_handler, types::boundary_id(0), Functions::ZeroFunction<dim>(), constraints);
  VectorTools::interpolate_boundary_values(
    dof_handler, types::boundary_id(1), Functions::ConstantFunction<dim>(1.0), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
}

template <int dim>
void Step9<dim>::assemble_system()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  const double      coef_difusion = 0.005;

  FEValues<dim> fe_values(fe,
                           quadrature_formula,
                           update_values | update_gradients | update_hessians |
                             update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>                   cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                       cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> advection_directions(n_q_points);
  AdvectionField<dim>         advection_field;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs    = 0;
      fe_values.reinit(cell);

      advection_field.value_list(fe_values.get_quadrature_points(),
                                 advection_directions);
      const double h = cell->diameter();

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const Tensor<1, dim> u_q    = advection_directions[q_point];
          const double         u_mag  = u_q.norm();

          double         tau = 0.0;
          Tensor<1, dim> u_unit;

          if (u_mag > 1e-10)
            {
              u_unit = u_q / u_mag;
              const double pe       = (u_mag * h) / (2.0 * coef_difusion);
              const double alpha_opt = 1.0 / std::tanh(pe) - (1.0 / pe);
              tau = (alpha_opt * h) / 2.0;
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += 0.0; // Fuente = 0
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Difusión estándar Galerkin
                  const double k_dif =
                    coef_difusion *
                    (fe_values.shape_grad(i, q_point) *
                     fe_values.shape_grad(j, q_point));

                  // Difusión SUPG: tau*(u_hat·∇W_i)*(-k*Δφ_j)
                  const double ks_dif =
                    tau *
                    (u_unit * fe_values.shape_grad(i, q_point)) *
                    (-coef_difusion) *
                    trace(fe_values.shape_hessian(j, q_point));

                  // Advección estándar Galerkin
                  const double k_adv =
                    fe_values.shape_value(i, q_point) *
                    (u_q * fe_values.shape_grad(j, q_point));

                  // Advección SUPG: tau*(u_hat·∇W_i)*(u·∇φ_j)
                  const double ks_adv =
                    tau *
                    (u_unit * fe_values.shape_grad(i, q_point)) *
                    (u_q * fe_values.shape_grad(j, q_point));

                  cell_matrix(i, j) +=
                    (k_dif + ks_dif + k_adv + ks_adv) * fe_values.JxW(q_point);
                }
            }
        }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

template <int dim>
void Step9<dim>::solve()
{
  SolverControl solver_control(
    std::max<std::size_t>(5000, system_rhs.size() * 2),
    1e-10 * system_rhs.l2_norm());

  SolverGMRES<Vector<double>> solver(
    solver_control, SolverGMRES<Vector<double>>::AdditionalData(50));

  // CORRECCIÓN 2: el campo correcto es extra_off_diagonals, no fill_in
  SparseILU<double> preconditioner;
  SparseILU<double>::AdditionalData ilu_data;
  ilu_data.extra_off_diagonals = 0; // ILU(0): sin llenado extra
  preconditioner.initialize(system_matrix, ilu_data);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  std::cout << "   Iterations required for convergence: "
            << solver_control.last_step() << '\n'
            << "   Norm of residual at convergence:     "
            << solver_control.last_value() << '\n';
}

template <int dim>
void Step9<dim>::refine_grid()
{
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  GradientEstimation::estimate(fe, dof_handler, solution, estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_number(
    triangulation, estimated_error_per_cell, refinamiento, coaserning);
  triangulation.execute_coarsening_and_refinement();
}

// ----------------------------------------------------------------
template <int dim>
void Step9<dim>::output_results(const unsigned int cycle) const
{
  // --- Malla ---
  {
    GridOut           grid_out;
    // CORRECCIÓN 3: write_vtu para la malla, consistente con extensión .vtu
    const std::string filename = generar_nombre_archivo(cycle, dim, "grid-");
    std::ofstream     output(filename);
    grid_out.write_vtu(triangulation, output);
    std::cout << "   Grid written to " << filename << std::endl;
  }

  // --- Solución ---
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(8);

    // CORRECCIÓN 4: write_vtu soporta compresión (write_vtk no lo hacía)
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);

    const std::string filename = generar_nombre_archivo(cycle, dim);
    std::ofstream     output(filename);
    data_out.write_vtu(output); // VTU: compresión funcional + formato moderno
    std::cout << "Output written to " << filename << std::endl;
  }
}

template <int dim>
void GradientEstimation::estimate(const FiniteElement<dim> &fe,
                                  const DoFHandler<dim>    &dof_handler,
                                  const Vector<double>     &solution,
                                  Vector<float>            &error_per_cell)
{
  Assert(error_per_cell.size() ==
           dof_handler.get_triangulation().n_active_cells(),
         ExcInvalidVectorLength(
           error_per_cell.size(),
           dof_handler.get_triangulation().n_active_cells()));

  FEValues<dim>        fe_midpoint_value(fe,
                                         QMidpoint<dim>(),
                                         update_values | update_quadrature_points);
  std::vector<double>  cell_midpoint_value(1);
  std::vector<double>  neighbor_midpoint_value(1);
  std::vector<typename DoFHandler<dim>::active_cell_iterator> active_neighbors;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      Tensor<2, dim> Y;
      fe_midpoint_value.reinit(cell);
      const Point<dim> this_center = fe_midpoint_value.quadrature_point(0);
      fe_midpoint_value.get_function_values(solution, cell_midpoint_value);

      active_neighbors.clear();
      for (const auto face_n : cell->face_indices())
        if (!cell->at_boundary(face_n))
          {
            const auto face     = cell->face(face_n);
            const auto neighbor = cell->neighbor(face_n);

            if (neighbor->is_active())
              active_neighbors.push_back(neighbor);
            else
              {
                if (dim == 1)
                  {
                    auto neighbor_child = neighbor;
                    while (neighbor_child->has_children())
                      neighbor_child = neighbor_child->child(face_n == 0 ? 1 : 0);
                    Assert(neighbor_child->neighbor(face_n == 0 ? 1 : 0) == cell,
                           ExcInternalError());
                    active_neighbors.push_back(neighbor_child);
                  }
                else
                  for (unsigned int subface_n = 0;
                       subface_n < face->n_children();
                       ++subface_n)
                    active_neighbors.push_back(
                      cell->neighbor_child_on_subface(face_n, subface_n));
              }
          }

      Tensor<1, dim> projected_gradient;

      for (const auto &neighbor : active_neighbors)
        {
          fe_midpoint_value.reinit(neighbor);
          const Point<dim> neighbor_center = fe_midpoint_value.quadrature_point(0);
          fe_midpoint_value.get_function_values(solution, neighbor_midpoint_value);

          Tensor<1, dim> y        = neighbor_center - this_center;
          const double   distance = y.norm();
          y /= distance;

          for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
              Y[i][j] += y[i] * y[j];

          projected_gradient +=
            (neighbor_midpoint_value[0] - cell_midpoint_value[0]) / distance * y;
        }

      AssertThrow(determinant(Y) != 0, ExcInsufficientDirections());

      const Tensor<2, dim> Y_inverse = invert(Y);
      const Tensor<1, dim> gradient  = Y_inverse * projected_gradient;

      error_per_cell(cell->active_cell_index()) =
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2) * gradient.norm();
    }
}

template <int dim>
void Step9<dim>::run()
{
  const unsigned int lim_cycle = (dim == 2) ? 15 : 4;

  for (unsigned int cycle = 0; cycle < lim_cycle; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;
      if (cycle == 0)
        make_grid();
      else
        refine_grid();

      std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells() << std::endl;
      setup_system();
      std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs() << std::endl;

      assemble_system();
      solve();
      output_results(cycle);
    }
}

int main()
{
  try
    {
      MultithreadInfo::set_thread_limit();
      Step9<2> laplace_problem_2d;
      laplace_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << "----------------------------------------------------" << std::endl
                << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << "----------------------------------------------------" << std::endl
                << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  return 0;
}