#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <filesystem>
#include <functional>
#include <map>
#include <vector>

namespace fs = std::filesystem;
using namespace dealii;

// ---------------------------------------------------------------------
//  VARIABLES GLOBALES
// ---------------------------------------------------------------------
const int         ref_2d = 2, ref_3d = 3;
const std::string base      = "solution_ref";
const std::string extension = ".vtu";
const float       refinamiento = 0.2, coaserning = 0.1;

// Esquema temporal
const double       theta_param               = 0.5;  // 0.5 Crank-Nicolson, 1.0 implícito
const double       t_inicial                 = 0.0;
const double       t_final                   = 4.0;
const double       delta_t                   = 1.0 / 500.0; // delta = 0.002
const unsigned int pasos_entre_refinamientos = 25;
const unsigned int pasos_entre_salidas       = 10;
const unsigned int niveles_extra_max         = 3; // niveles de refinamiento por encima de la malla inicial

std::string generar_nombre_archivo(const int          cycle,
                                   const int          dimension,
                                   const std::string &prefix = "")
{
  std::string folder = "results_" + std::to_string(dimension) + "d";
  fs::create_directories(folder);
  return folder + "/" + prefix + base + std::to_string(cycle) + extension;
}

// ---------------------------------------------------------------------
//  Campo de advección u = (-y, x)
// ---------------------------------------------------------------------
template <int dim>
class AdvectionField : public TensorFunction<1, dim>
{
public:
  virtual Tensor<1, dim> value(const Point<dim> &p) const override
  {
    Tensor<1, dim> value;
    value[0] = -p[1];
    value[1] = p[0];
    if (dim == 3)
      value[2] = 0.0;
    return value;
  }
};

// ---------------------------------------------------------------------
//  Estimador de gradiente en paralelo
// ---------------------------------------------------------------------
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

private:
  template <int dim>
  struct EstimateScratchData
  {
    EstimateScratchData(const FiniteElement<dim> &fe,
                        const Vector<double>     &solution,
                        Vector<float>            &error_per_cell)
      : fe_midpoint_value(fe,
                          QMidpoint<dim>(),
                          update_values | update_quadrature_points)
      , solution(solution)
      , error_per_cell(error_per_cell)
      , cell_midpoint_value(1)
      , neighbor_midpoint_value(1)
    {
      active_neighbors.reserve(GeometryInfo<dim>::faces_per_cell *
                               GeometryInfo<dim>::max_children_per_face);
    }

    EstimateScratchData(const EstimateScratchData &scratch_data)
      : fe_midpoint_value(scratch_data.fe_midpoint_value.get_fe(),
                          scratch_data.fe_midpoint_value.get_quadrature(),
                          update_values | update_quadrature_points)
      , solution(scratch_data.solution)
      , error_per_cell(scratch_data.error_per_cell)
      , cell_midpoint_value(1)
      , neighbor_midpoint_value(1)
    {}

    FEValues<dim>                                               fe_midpoint_value;
    std::vector<typename DoFHandler<dim>::active_cell_iterator> active_neighbors;
    const Vector<double> &solution;
    Vector<float>        &error_per_cell;
    std::vector<double>   cell_midpoint_value;
    std::vector<double>   neighbor_midpoint_value;
  };

  // Cada celda escribe solo en SU entrada de error_per_cell -> no hay conflicto entre hilos, así que no se necesita fase de copia.
  struct EstimateCopyData
  {};

  template <int dim>
  static void
  estimate_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                EstimateScratchData<dim> &scratch_data,
                const EstimateCopyData   &copy_data);
};

template <int dim>
class Step9Transient
{
public:
  Step9Transient();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_matrices(); // M y A en paralelo, una vez por malla
  void do_time_step();
  void solve();
  void refine_mesh();
  void output_results();

  struct AssemblyScratchData
  {
    AssemblyScratchData(const FiniteElement<dim> &fe)
      : fe_values(fe,
                  QGauss<dim>(fe.degree + 1),
                  update_values | update_gradients | update_hessians |
                    update_quadrature_points | update_JxW_values)
      , advection_directions(fe_values.get_quadrature().size())
    {}

    AssemblyScratchData(const AssemblyScratchData &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , advection_directions(scratch_data.advection_directions.size())
    {}

    FEValues<dim>               fe_values;
    std::vector<Tensor<1, dim>> advection_directions;
    AdvectionField<dim>         advection_field;
  };

  struct AssemblyCopyData
  {
    FullMatrix<double>                   cell_mass_matrix;
    FullMatrix<double>                   cell_laplace_matrix;
    std::vector<types::global_dof_index> local_dof_indices;
  };

  void local_assemble_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData                                  &scratch_data,
    AssemblyCopyData                                     &copy_data);

  void copy_local_to_global(const AssemblyCopyData &copy_data);

  // ----- Datos -----
  Triangulation<dim>        triangulation;
  const FE_Q<dim>           fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints; // solo hanging nodes (Dirichlet se aplica por paso)

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> mass_matrix;    // M (Galerkin + SUPG)
  SparseMatrix<double> laplace_matrix; // A (difusión + advección + SUPG)
  SparseMatrix<double> system_matrix;  // M + theta*dt*A

  Vector<double> solution;     // phi^{n+1}
  Vector<double> old_solution; // phi^n
  Vector<double> system_rhs;

  double       time;
  double       time_step;
  unsigned int timestep_number;
  const double theta;
  const double coef_difusion = 0.000005;

  unsigned int min_grid_level;
  unsigned int max_grid_level;

  std::vector<std::pair<double, std::string>> times_and_names; // para el .pvd
  TimerOutput                                 computing_timer;
};

template <int dim>
Step9Transient<dim>::Step9Transient()
  : fe(3)
  , dof_handler(triangulation)
  , time(t_inicial)
  , time_step(delta_t)
  , timestep_number(0)
  , theta(theta_param)
  , min_grid_level(dim == 2 ? ref_2d : ref_3d)
  , max_grid_level((dim == 2 ? ref_2d : ref_3d) + niveles_extra_max)
  , computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{}

// ---------------------------------------------------------------------
//  Geometría en L (sin cambios)
// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::make_grid()
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
  triangulation.refine_global(min_grid_level);

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

// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  // Como en Step 26: 'constraints' guarda SOLO los hanging nodes.
  // Dirichlet se aplica en cada paso con apply_boundary_values.
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
  laplace_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

// ---------------------------------------------------------------------
//  Ensamble paralelo
// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::assemble_matrices()
{
  TimerOutput::Scope t(computing_timer, "ensamble (WorkStream)");

  mass_matrix    = 0;
  laplace_matrix = 0;

  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &Step9Transient::local_assemble_system,
                  &Step9Transient::copy_local_to_global,
                  AssemblyScratchData(fe),
                  AssemblyCopyData());
}

// Trabajo local: se ejecuta en paralelo, una celda por tarea.
// Solo escribe en scratch_data y copy_data (propios de cada hilo).
template <int dim>
void Step9Transient<dim>::local_assemble_system(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  AssemblyScratchData                                  &scratch_data,
  AssemblyCopyData                                     &copy_data)
{
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points =
    scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.cell_laplace_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);

  FEValues<dim> &fe_values = scratch_data.fe_values;
  fe_values.reinit(cell);

  scratch_data.advection_field.value_list(fe_values.get_quadrature_points(),
                                          scratch_data.advection_directions);

  const double h = cell->diameter();

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const Tensor<1, dim> u_q   = scratch_data.advection_directions[q];
      const double         u_mag = u_q.norm();

      // tau SUPG transitorio (unidades de tiempo)
      const double term_t = 2.0 / time_step;
      const double term_a = 2.0 * u_mag / h;
      const double term_d = 4.0 * coef_difusion / (h * h);
      const double tau =
        1.0 / std::sqrt(term_t * term_t + term_a * term_a + term_d * term_d);

      const double JxW = fe_values.JxW(q);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const double         w_i      = fe_values.shape_value(i, q);
          const Tensor<1, dim> grad_w_i = fe_values.shape_grad(i, q);
          // Función de peso SUPG: tau * (u · ∇w_i)
          const double supg_i = tau * (u_q * grad_w_i);

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const double         phi_j      = fe_values.shape_value(j, q);
              const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);
              const double lap_phi_j = trace(fe_values.shape_hessian(j, q));

              // M: Galerkin + SUPG (término dphi/dt del residuo)
              copy_data.cell_mass_matrix(i, j) +=
                (w_i * phi_j + supg_i * phi_j) * JxW;

              // A: difusión + advección Galerkin + SUPG
              const double k_dif  = coef_difusion * (grad_w_i * grad_phi_j);
              const double k_adv  = w_i * (u_q * grad_phi_j);
              const double ks_dif = supg_i * (-coef_difusion * lap_phi_j);
              const double ks_adv = supg_i * (u_q * grad_phi_j);

              copy_data.cell_laplace_matrix(i, j) +=
                (k_dif + k_adv + ks_dif + ks_adv) * JxW;
            }
        }
    }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

// Copia a las matrices globales: WorkStream la ejecuta en serie,
// así que no hay condiciones de carrera al escribir en M y A.
template <int dim>
void Step9Transient<dim>::copy_local_to_global(const AssemblyCopyData &copy_data)
{
  mass_matrix.add(copy_data.local_dof_indices, copy_data.cell_mass_matrix);
  laplace_matrix.add(copy_data.local_dof_indices,
                     copy_data.cell_laplace_matrix);
}

// ---------------------------------------------------------------------
//  Paso de tiempo (theta-scheme)
//  (M + theta*dt*A) phi^{n+1} = (M - (1-theta)*dt*A) phi^n     (f = 0)
// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::do_time_step()
{
  {
    TimerOutput::Scope t(computing_timer, "armado sistema theta");

    Vector<double> tmp(solution.size());

    mass_matrix.vmult(system_rhs, old_solution);
    laplace_matrix.vmult(tmp, old_solution);
    system_rhs.add(-(1.0 - theta) * time_step, tmp);

    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * time_step, laplace_matrix);

    constraints.condense(system_matrix, system_rhs);

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             types::boundary_id(0),
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);
    VectorTools::interpolate_boundary_values(
      dof_handler,
      types::boundary_id(1),
      Functions::ConstantFunction<dim>(1.0),
      boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }

  solve();
}

template <int dim>
void Step9Transient<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "solver (GMRES+ILU)");

  SolverControl solver_control(std::max<std::size_t>(5000,
                                                     system_rhs.size() * 2),
                               1e-10 * system_rhs.l2_norm());
  SolverGMRES<Vector<double>> solver(
    solver_control, SolverGMRES<Vector<double>>::AdditionalData(50));

  SparseILU<double>                 preconditioner;
  SparseILU<double>::AdditionalData ilu_data;
  ilu_data.extra_off_diagonals = 0;
  preconditioner.initialize(system_matrix, ilu_data);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  std::cout << "   GMRES: " << solver_control.last_step() << " iteraciones"
            << std::endl;
}

// ---------------------------------------------------------------------
//  AMR + transferencia de la solución
// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::refine_mesh()
{
  TimerOutput::Scope t(computing_timer, "refinamiento + transfer");

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  GradientEstimation::estimate(fe,
                               dof_handler,
                               solution,
                               estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                  estimated_error_per_cell,
                                                  refinamiento,
                                                  coaserning);

  // Límites de nivel (como en Step 26) para que la malla no crezca sin control
  if (triangulation.n_levels() > max_grid_level)
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(max_grid_level))
      cell->clear_refine_flag();
  for (const auto &cell :
       triangulation.active_cell_iterators_on_level(min_grid_level))
    cell->clear_coarsen_flag();

  SolutionTransfer<dim, Vector<double>> solution_trans(dof_handler);
  Vector<double>                        previous_solution = solution;

  triangulation.prepare_coarsening_and_refinement();
  solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
  triangulation.execute_coarsening_and_refinement();

  setup_system();

#if DEAL_II_VERSION_GTE(9, 6, 0)
  solution_trans.interpolate(solution); // interfaz nueva (9.6+)
#else
  solution_trans.interpolate(previous_solution, solution);
#endif
  constraints.distribute(solution);
  old_solution = solution;
}

// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "salida VTU");

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches(fe.degree);

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  vtk_flags.time              = time;
  vtk_flags.cycle             = timestep_number;
  data_out.set_flags(vtk_flags);

  const std::string filename = generar_nombre_archivo(timestep_number, dim);
  std::ofstream     output(filename);
  data_out.write_vtu(output);

  // Archivo .pvd: abrir ESTE en ParaView para tener la serie temporal
  times_and_names.emplace_back(time, fs::path(filename).filename().string());
  std::ofstream pvd_output("results_" + std::to_string(dim) + "d/solution.pvd");
  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

// ---------------------------------------------------------------------
//  Estimador de gradiente: implementación
// ---------------------------------------------------------------------
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

  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  &GradientEstimation::template estimate_cell<dim>,
                  std::function<void(const EstimateCopyData &)>(),
                  EstimateScratchData<dim>(fe, solution, error_per_cell),
                  EstimateCopyData());
}

template <int dim>
void GradientEstimation::estimate_cell(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  EstimateScratchData<dim>                             &scratch_data,
  const EstimateCopyData &)
{
  Tensor<2, dim> Y;

  scratch_data.fe_midpoint_value.reinit(cell);
  const Point<dim> this_center =
    scratch_data.fe_midpoint_value.quadrature_point(0);
  scratch_data.fe_midpoint_value.get_function_values(
    scratch_data.solution, scratch_data.cell_midpoint_value);

  scratch_data.active_neighbors.clear();
  for (const auto face_n : cell->face_indices())
    if (!cell->at_boundary(face_n))
      {
        const auto face     = cell->face(face_n);
        const auto neighbor = cell->neighbor(face_n);

        if (neighbor->is_active())
          scratch_data.active_neighbors.push_back(neighbor);
        else
          {
            if (dim == 1)
              {
                auto neighbor_child = neighbor;
                while (neighbor_child->has_children())
                  neighbor_child = neighbor_child->child(face_n == 0 ? 1 : 0);
                Assert(neighbor_child->neighbor(face_n == 0 ? 1 : 0) == cell,
                       ExcInternalError());
                scratch_data.active_neighbors.push_back(neighbor_child);
              }
            else
              for (unsigned int subface_n = 0; subface_n < face->n_children();
                   ++subface_n)
                scratch_data.active_neighbors.push_back(
                  cell->neighbor_child_on_subface(face_n, subface_n));
          }
      }

  Tensor<1, dim> projected_gradient;
  for (const auto &neighbor : scratch_data.active_neighbors)
    {
      scratch_data.fe_midpoint_value.reinit(neighbor);
      const Point<dim> neighbor_center =
        scratch_data.fe_midpoint_value.quadrature_point(0);
      scratch_data.fe_midpoint_value.get_function_values(
        scratch_data.solution, scratch_data.neighbor_midpoint_value);

      Tensor<1, dim> y        = neighbor_center - this_center;
      const double   distance = y.norm();
      y /= distance;

      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          Y[i][j] += y[i] * y[j];

      projected_gradient += (scratch_data.neighbor_midpoint_value[0] -
                             scratch_data.cell_midpoint_value[0]) /
                            distance * y;
    }

  AssertThrow(determinant(Y) != 0, ExcInsufficientDirections());

  const Tensor<2, dim> Y_inverse = invert(Y);
  const Tensor<1, dim> gradient  = Y_inverse * projected_gradient;

  // Cada celda escribe solo en su propia posición -> seguro entre hilos
  scratch_data.error_per_cell(cell->active_cell_index()) =
    std::pow(cell->diameter(), 1 + 1.0 * dim / 2) * gradient.norm();
}

// ---------------------------------------------------------------------
//  Bucle temporal
// ---------------------------------------------------------------------
template <int dim>
void Step9Transient<dim>::run()
{
  std::cout << "Hilos disponibles: " << MultithreadInfo::n_threads()
            << std::endl;

  make_grid();
  setup_system();

  // Condición inicial phi(x,0) = 0
  VectorTools::interpolate(dof_handler,
                           Functions::ZeroFunction<dim>(),
                           old_solution);
  constraints.distribute(old_solution);
  solution = old_solution;

  assemble_matrices();

  std::cout << "Malla inicial: " << triangulation.n_active_cells()
            << " celdas, " << dof_handler.n_dofs() << " DoFs" << std::endl;

  output_results();

  while (time < t_final - 1e-12)
    {
      time += time_step;
      ++timestep_number;
      std::cout << "Paso " << timestep_number << ", t = " << time << std::endl;

      do_time_step();

      if (timestep_number % pasos_entre_refinamientos == 0)
        {
          refine_mesh();
          assemble_matrices(); // M y A dependen de la malla
          std::cout << "   AMR -> " << triangulation.n_active_cells()
                    << " celdas, " << dof_handler.n_dofs() << " DoFs"
                    << std::endl;
        }

      if (timestep_number % pasos_entre_salidas == 0)
        output_results();

      old_solution = solution;
    }
}

int main()
{
  try
    {
      // Sin argumento: usa todos los núcleos (o DEAL_II_NUM_THREADS si está definida)
      MultithreadInfo::set_thread_limit(8);

      Step9Transient<2> problema_transitorio_2d;
      problema_transitorio_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << "----------------------------------------------------"
                << std::endl
                << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << "----------------------------------------------------"
                << std::endl
                << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}