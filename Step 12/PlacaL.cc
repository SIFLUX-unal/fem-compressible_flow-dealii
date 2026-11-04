// ==========================================================
// Solución de problema de convección - difusión usando elementos finitos discontinuos DGFEM
// ==========================================================
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/meshworker/mesh_loop.h> 
#include <deal.II/base/table.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>



namespace fs = std::filesystem;


// ==========================================================
// Variables Globales
// ==========================================================
constexpr int     dim = 2;
const int         ref_2d = 5, ref_3d = 2;
const std::string base      = "solution_ref";
const std::string extension = ".vtu"; // usar data_out.write_vtu()
const double      refinamiento = 0.3, coarsening = 0.1;
const unsigned int n_ciclos = 5;

// Problema
const double epsilon = 0.005; // coeficiente de difusión

// Método DG
const unsigned int grado_fe          = 1;    // grado polinomial p (FE_DGQ)
const double       theta_dg          = 0.0; // -1: SIPG, +1: NIPG, 0: IIPG
const double       coef_penalizacion = 100; // C en sigma = C p(p+1)/h (parámetro de penalidad)

// Solver
const unsigned int gmres_max_iter = 5000;
const double       gmres_tol_rel  = 1e-8;
const unsigned int gmres_base     = 100;

std::string generar_nombre_archivo(const int          cycle,
                                   const int          dimension,
                                   const std::string &prefix = "")
{
  std::string folder = "results_" + std::to_string(dimension) + "d";
  fs::create_directories(folder);
  return folder + "/" + prefix + base + std::to_string(cycle) + extension;
}



namespace ProblemaDG
{
  using namespace dealii;

  // ==========================================================
  // CLASE BoundaryValues
  // ==========================================================
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double>           &values,
                            const unsigned int component = 0) const override;
  };

  // Especificación de los valores de las condiciones de contorno
  // phi = 1 en el tramo inferior (y = 0, x > 0.5), phi = 0 en el resto
  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double>           &values,
                                       const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    AssertDimension(values.size(), points.size());

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        if (std::abs(points[i][1]) < 1e-10 && points[i][0] > 0.5)
          values[i] = 1.;
        else
          values[i] = 0.;
      }
  }


  // Definición del campo de velocidad beta
  template <int dim>
  Tensor<1, dim> beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());

    Tensor<1, dim> wind_field;
    wind_field[0] = -p[1];
    wind_field[1] =  p[0];

    return wind_field; // sin normalizar, como en el enunciado
  }



  // ==========================================================
  // Scracth_data
  // ==========================================================
  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim>             &mapping,
                const FiniteElement<dim>       &fe,
                const Quadrature<dim>          &quadrature,
                const Quadrature<dim - 1>      &quadrature_face,
                const UpdateFlags              update_flags = update_values |
                                                              update_gradients |
                                                              update_quadrature_points |
                                                              update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, quadrature, update_flags)
      , fe_interface_values(mapping, fe, quadrature_face, interface_update_flags)
    {}

    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                            scratch_data.fe_interface_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };


  // ==========================================================
  // Copy_data
  // ==========================================================
  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };

  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template <class Iterator>
     void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      face_data.clear(); // evita arrastrar caras de la celda anterior
    }
  };


  // ==========================================================
  // DECLARACIÓN DE LA CLASE PRINCIPAL Problema1
  // ==========================================================
  template <int dim>
  class Problema1
  {
  public:
    Problema1();
    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim> triangulation;
    const MappingQ1<dim> mapping;

    const FE_DGQ<dim> fe;
    DoFHandler<dim> dof_handler;

    const QGauss<dim>  quadrature;
    const QGauss<dim -1> quadrature_face;

    SparsityPattern  sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> right_hand_side;

  };


  // ==========================================================
  // CONSTRUCTOR
  // ==========================================================
  template <int dim>
  Problema1<dim>::Problema1()
    : mapping()
    , fe(grado_fe)
    , dof_handler(triangulation)
    , quadrature(fe.tensor_degree() + 1)
    , quadrature_face(fe.tensor_degree() + 1)
  {}



  // ==========================================================
  // make_grid()
  // ==========================================================

  template <int dim>
  void Problema1<dim>::make_grid()
  {
    // Cuadrado [0,1]^2 dividido en 2x2 celdas de lado 0.5
    const std::vector<std::vector<double>> pasos = {{0.5, 0.5}, {0.5, 0.5}};

    // material_id por celda: mat[i][j], con i en x y j en y
    Table<2, types::material_id> mat(2, 2);
    mat[0][0] = numbers::invalid_material_id; // celda inferior izquierda: se elimina
    mat[1][0] = 0;
    mat[0][1] = 0;
    mat[1][1] = 0;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              pasos,
                                              Point<dim>(0., 0.),
                                              mat);

    triangulation.refine_global(ref_2d);
  }



  // ==========================================================
  // setup_system
  // ==========================================================
  template <int dim>
  void Problema1<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "=========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    // Patrón de dispersión para DG: incluye el acoplamiento entre
    // celdas vecinas a través de las caras
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp); // construcción del patron de dispersión para los elementos finitos discontinuos
    sparsity_pattern.copy_from(dsp);

    // El ensamble se hace con el mesh_loop
    // COnfiguración de la matriz de rígidez global y vectores globales
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
  }


  // ==========================================================
  // assemble_system()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::assemble_system()
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;
    const BoundaryValues<dim> boundary_function;

    // ---------------- Bloque para las integrales dentro de las celdas ----------------
    const auto cell_worker = [&](const Iterator   &cell,
                                 ScratchData<dim> &scratch_data,
                                 CopyData         &copy_data) {
      const unsigned int n_dofs =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);

      const FEValues<dim>       &fe_v     = scratch_data.fe_values;
      const auto                &q_points = fe_v.get_quadrature_points();
      const std::vector<double> &JxW      = fe_v.get_JxW_values();

      // f = 0, no hay término fuente
      for (unsigned int q = 0; q < fe_v.n_quadrature_points; ++q)
        {
          const auto b = beta(q_points[q]);
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              // A_K(i,j) += −(β·∇ω_i) φ_j · w_q|J|
              copy_data.cell_matrix(i, j) +=
                -b * fe_v.shape_grad(i, q) * fe_v.shape_value(j, q) * JxW[q];
        }
    };

    // ---------------- Caras de borde del dominio (no compartidas entre celdas) ----------------
    // g -> dato de la condición de borde de dirchlecht
    const auto boundary_worker = [&](const Iterator     &cell,
                                     const unsigned int &face_no,
                                     ScratchData<dim>   &scratch_data,
                                     CopyData           &copy_data) {
      scratch_data.fe_interface_values.reinit(cell, face_no);
      const FEFaceValuesBase<dim> &fe_face =
        scratch_data.fe_interface_values.get_fe_face_values(0);

      const auto &q_points = fe_face.get_quadrature_points();
      const auto &JxW      = fe_face.get_JxW_values();
      const auto &normals  = fe_face.get_normal_vectors();

      const unsigned int n_dofs = fe_face.get_fe().n_dofs_per_cell();

      std::vector<double> g(q_points.size());
      boundary_function.value_list(q_points, g);

      for (unsigned int q = 0; q < q_points.size(); ++q)
        {
          const double beta_n = beta(q_points[q]) * normals[q];

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double w = fe_face.shape_value(i, q);

              if (beta_n > 0) // outflow: va a la matriz, (β·n) φ_j ω_i
                for (unsigned int j = 0; j < n_dofs; ++j)
                  copy_data.cell_matrix(i, j) +=
                    fe_face.shape_value(j, q) * w * beta_n * JxW[q];
              else            // inflow: va al RHS, −(β·n) g ω_i
                copy_data.cell_rhs(i) += -w * g[q] * beta_n * JxW[q];
            }
        }
    };
    // ---------------- BLoque para caras compartidas entre celdas ----------------
    const auto face_worker = [&](const Iterator     &cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const Iterator     &ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 ScratchData<dim>   &scratch_data,
                                 CopyData           &copy_data) {
      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);

      const auto &q_points = fe_iv.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &cdf = copy_data.face_data.back();

      const unsigned int n_dofs = fe_iv.n_current_interface_dofs();
      cdf.joint_dof_indices     = fe_iv.get_interface_dof_indices();
      cdf.cell_matrix.reinit(n_dofs, n_dofs);

      const auto &JxW     = fe_iv.get_JxW_values();
      const auto &normals = fe_iv.get_normal_vectors();

      for (unsigned int q = 0; q < q_points.size(); ++q)
        {
          const double beta_n = beta(q_points[q]) * normals[q];

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const double jump_w = fe_iv.jump_in_shape_values(i, q);

              for (unsigned int j = 0; j < n_dofs; ++j)
                // (β·n) φ_j^up [ω_i]
                cdf.cell_matrix(i, j) +=
                  jump_w * fe_iv.shape_value((beta_n > 0), j, q) * beta_n *
                  JxW[q];
            }
        }
    };

    // ---------------- Copia local -> global ----------------
    const AffineConstraints<double> constraints; // vacío en DG

    const auto copier = [&](const CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             right_hand_side);
      for (const auto &cdf : c.face_data)
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                               cdf.joint_dof_indices,
                                               system_matrix);
    };

    ScratchData<dim> scratch_data(mapping, fe, quadrature, quadrature_face);
    CopyData         copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }


  // ==========================================================
  // solve()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::solve()
  {
    SolverControl solver_control(gmres_max_iter,
                                 gmres_tol_rel * right_hand_side.l2_norm());

    SolverGMRES<Vector<double>>::AdditionalData additional_data;
    additional_data.max_basis_size = gmres_base;
    SolverGMRES<Vector<double>> solver(solver_control, additional_data);

    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());

    solver.solve(system_matrix, solution, right_hand_side, preconditioner);

    std::cout << "  Solver converged in " << solver_control.last_step()
              << " iterations." << std::endl;
  }





  // ==========================================================
  // Refine_grid()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::refine_grid()
  {
    // Gradientes aproximados en precisión float: solo sirven de indicador
    Vector<float> gradient_indicator(triangulation.n_active_cells());

    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof_handler,
                                                  solution,
                                                  gradient_indicator);

    // Escalado por celda con h^{1+d/2}
    unsigned int cell_no = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      gradient_indicator(cell_no++) *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    gradient_indicator,
                                                    refinamiento,
                                                    coarsening);

    triangulation.execute_coarsening_and_refinement();
  }





  // ==========================================================
  // output_results()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::output_results(const unsigned int cycle) const
  {
    const std::string filename = generar_nombre_archivo(cycle, dim);
    std::cout << "  Writing solution to <" << filename << '>' << std::endl;
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);

    data_out.build_patches(mapping);

    data_out.write_vtu(output);

    {
      Vector<float> values(triangulation.n_active_cells());
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        Functions::ZeroFunction<dim>(),
                                        values,
                                        quadrature,
                                        VectorTools::Linfty_norm);
      const double l_infty = VectorTools::compute_global_error(triangulation,
                                                                values,
                                                                VectorTools::Linfty_norm);
      std::cout << "  L-infinity norm: " << l_infty << std::endl;
    }
  }





  // ==========================================================
  // run()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < n_ciclos; ++cycle)
      {
        std::cout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          make_grid();
        else
          refine_grid();

        std::cout << "  Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        setup_system();

        std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        assemble_system();
        solve();

        output_results(cycle);
      }
  }

}




// ==========================================================
// MAIN
// ==========================================================
int main()
{
  try
    {
      ProblemaDG::Problema1<dim> problema;
      problema.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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