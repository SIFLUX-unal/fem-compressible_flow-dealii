// ===============================================================================================================
// Solución de problema de convección - difusión usando elementos finitos discontinuos DGFEM en estado transitorio
// ===============================================================================================================


// ---------------- Base ----------------
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/data_out_base.h>          // VtkFlags y write_pvd_record

// ---------------- Álgebra lineal ----------------
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>      // vacío en DG, lo usa el copier
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>      // SSOR por bloques y M^{-1} exacta
#include <deal.II/base/multithread_info.h>

// ---------------- Malla ----------------
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>

// ---------------- Elementos finitos y DoFs ----------------
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>                   // reemplaza fe_q.h del step 26
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// ---------------- Numerics ----------------
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_creator.h>     // create_mass_matrix
#include <deal.II/numerics/solution_transfer.h>  // AMR en el tiempo
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/meshworker/mesh_loop.h>

// ---------------- Estándar ----------------
#include <array>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <limits>  // Usado en el bloque de compute_time_step


namespace fs = std::filesystem;


// ==========================================================
// Variables Globales
// ==========================================================
constexpr int dim = 2;

// ---------------- Malla ----------------
const unsigned int refinamiento_inicial = 4;
const bool         usar_amr             = false;
const unsigned int n_pre_refinamientos  = 2;
const unsigned int nivel_min            = refinamiento_inicial;
const unsigned int nivel_max            = refinamiento_inicial + n_pre_refinamientos;
const unsigned int pasos_entre_refin    = 5;
const double       frac_refinar         = 0.3;
const double       frac_coarsen         = 0.1;

// ---------------- Método DG ----------------
const unsigned int grado_fe = 1;

// ---------------- Solver (esquema theta) ----------------
const unsigned int gmres_max_iter = 5000;
const double       gmres_tol_rel  = 1e-8;
const unsigned int gmres_base     = 100;

// ---------------- Esquema temporal ----------------
enum class EsquemaTemporal { Theta, RungeKutta };
const EsquemaTemporal esquema     = EsquemaTemporal::RungeKutta;
const double          theta_valor = 0.5;

enum class TablaRK { Euler1, Heun2, Ralston2, Kutta3, SSP3, RK4, RK4_38 };
const TablaRK tabla_rk = TablaRK::RK4;

// ---------------- Tiempo ----------------
const double       t_inicial   = 0.0;
const double       t_final     = 2.0;
const unsigned int salida_cada = 10;

enum class PoliticaPaso { Fijo, CFL };
const PoliticaPaso politica_paso = PoliticaPaso::CFL;
const double       dt_fijo = 1.0 / 500;
const double       cfl     = 0.2;

// ---------------- Salida ----------------
const std::string carpeta_base = "results_";
const std::string base         = "solution_t";
const std::string extension    = ".vtu";



// ==========================================================
// namespace TablasRK: tablas de Butcher explícitas
// ==========================================================
namespace TablasRK
{
  using namespace dealii;

  // ----------------------------------------------------------
  // Contenedor de una tabla de Butcher explícita
  // ----------------------------------------------------------
  struct TablaButcher
  {
    unsigned int       etapas; // s
    unsigned int       orden;  // orden clásico del método
    std::string        nombre;
    FullMatrix<double> A;      // s x s, estrictamente triangular inferior
    Vector<double>     b;      // pesos del avance final
    Vector<double>     c;      // nodos temporales de cada etapa

    TablaButcher(const unsigned int s,
                 const unsigned int p,
                 const std::string &n)
      : etapas(s)
      , orden(p)
      , nombre(n)
      , A(s, s)
      , b(s)
      , c(s)
    {} // A, b y c quedan inicializados en cero
  };


  // ----------------------------------------------------------
  // Tablas disponibles (índices desde 0: A(1,0) es a_21)
  // ----------------------------------------------------------
  inline TablaButcher obtener_tabla(const ::TablaRK tipo)
  {
    switch (tipo)
      {
        case ::TablaRK::Euler1:
          {
            TablaButcher T(1, 1, "Euler explícito");
            T.b(0) = 1.;
            return T;
          }

        case ::TablaRK::Heun2:
          {
            TablaButcher T(2, 2, "Heun (RK2)");
            T.c(1)    = 1.;
            T.A(1, 0) = 1.;
            T.b(0) = 1. / 2;
            T.b(1) = 1. / 2;
            return T;
          }

        case ::TablaRK::Ralston2:
          {
            TablaButcher T(2, 2, "Ralston (RK2)");
            T.c(1)    = 2. / 3;
            T.A(1, 0) = 2. / 3;
            T.b(0) = 1. / 4;
            T.b(1) = 3. / 4;
            return T;
          }

        case ::TablaRK::Kutta3:
          {
            TablaButcher T(3, 3, "Kutta (RK3)");
            T.c(1) = 1. / 2;
            T.c(2) = 1.;
            T.A(1, 0) = 1. / 2;
            T.A(2, 0) = -1.;
            T.A(2, 1) = 2.;
            T.b(0) = 1. / 6;
            T.b(1) = 2. / 3;
            T.b(2) = 1. / 6;
            return T;
          }

        case ::TablaRK::SSP3: // Shu-Osher escrito en forma de Butcher
          {
            TablaButcher T(3, 3, "SSP-RK3 (Shu-Osher)");
            T.c(1) = 1.;
            T.c(2) = 1. / 2;
            T.A(1, 0) = 1.;
            T.A(2, 0) = 1. / 4;
            T.A(2, 1) = 1. / 4;
            T.b(0) = 1. / 6;
            T.b(1) = 1. / 6;
            T.b(2) = 2. / 3;
            return T;
          }

        case ::TablaRK::RK4:
          {
            TablaButcher T(4, 4, "RK4 clásico");
            T.c(1) = 1. / 2;
            T.c(2) = 1. / 2;
            T.c(3) = 1.;
            T.A(1, 0) = 1. / 2;
            T.A(2, 1) = 1. / 2;
            T.A(3, 2) = 1.;
            T.b(0) = 1. / 6;
            T.b(1) = 1. / 3;
            T.b(2) = 1. / 3;
            T.b(3) = 1. / 6;
            return T;
          }

        case ::TablaRK::RK4_38:
          {
            TablaButcher T(4, 4, "RK4 regla 3/8");
            T.c(1) = 1. / 3;
            T.c(2) = 2. / 3;
            T.c(3) = 1.;
            T.A(1, 0) = 1. / 3;
            T.A(2, 0) = -1. / 3;
            T.A(2, 1) = 1.;
            T.A(3, 0) = 1.;
            T.A(3, 1) = -1.;
            T.A(3, 2) = 1.;
            T.b(0) = 1. / 8;
            T.b(1) = 3. / 8;
            T.b(2) = 3. / 8;
            T.b(3) = 1. / 8;
            return T;
          }
      }

    AssertThrow(false, ExcMessage("Tabla RK no implementada."));
    return TablaButcher(1, 1, "");
  }


  // ----------------------------------------------------------
  // Verificación de las condiciones de orden
  // ----------------------------------------------------------
  // Comprueba consistencia (c_i = sum_j a_ij) y las condiciones de orden
  // hasta T.orden. Lanza excepción si alguna falla.
  inline void verificar_tabla(const TablaButcher &T, const double tol = 1e-13)
  {
    const unsigned int s = T.etapas;

    auto chequear = [&](const double       valor,
                        const double       esperado,
                        const std::string &cond) {
      AssertThrow(std::abs(valor - esperado) < tol,
                  ExcMessage("Tabla '" + T.nombre +
                             "' no cumple la condición: " + cond));
    };

    // Orden 1: sum b = 1, y consistencia c_i = sum_j a_ij
    double sb = 0.;
    for (unsigned int i = 0; i < s; ++i)
      {
        sb += T.b(i);
        double fila = 0.;
        for (unsigned int j = 0; j < s; ++j)
          fila += T.A(i, j);
        chequear(fila, T.c(i), "c_i = sum_j a_ij");
      }
    chequear(sb, 1., "sum b_i = 1");

    // Orden 2: sum b_i c_i = 1/2
    if (T.orden >= 2)
      {
        double v = 0.;
        for (unsigned int i = 0; i < s; ++i)
          v += T.b(i) * T.c(i);
        chequear(v, 1. / 2, "sum b_i c_i = 1/2");
      }

    // Orden 3: sum b c^2 = 1/3 ; sum b_i a_ij c_j = 1/6
    if (T.orden >= 3)
      {
        double v1 = 0., v2 = 0.;
        for (unsigned int i = 0; i < s; ++i)
          {
            v1 += T.b(i) * T.c(i) * T.c(i);
            for (unsigned int j = 0; j < s; ++j)
              v2 += T.b(i) * T.A(i, j) * T.c(j);
          }
        chequear(v1, 1. / 3, "sum b_i c_i^2 = 1/3");
        chequear(v2, 1. / 6, "sum b_i a_ij c_j = 1/6");
      }

    // Orden 4: cuatro condiciones
    if (T.orden >= 4)
      {
        double v1 = 0., v2 = 0., v3 = 0., v4 = 0.;
        for (unsigned int i = 0; i < s; ++i)
          {
            v1 += T.b(i) * std::pow(T.c(i), 3);
            for (unsigned int j = 0; j < s; ++j)
              {
                v2 += T.b(i) * T.c(i) * T.A(i, j) * T.c(j);
                v3 += T.b(i) * T.A(i, j) * T.c(j) * T.c(j);
                for (unsigned int l = 0; l < s; ++l)
                  v4 += T.b(i) * T.A(i, j) * T.A(j, l) * T.c(l);
              }
          }
        chequear(v1, 1. / 4,  "sum b_i c_i^3 = 1/4");
        chequear(v2, 1. / 8,  "sum b_i c_i a_ij c_j = 1/8");
        chequear(v3, 1. / 12, "sum b_i a_ij c_j^2 = 1/12");
        chequear(v4, 1. / 24, "sum b_i a_ij a_jl c_l = 1/24");
      }
  }
} // namespace TablasRK


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

  // phi = 1 en el tramo inferior (y = 0, x > 0.5), phi = 0 en el resto.
  // Hoy g no depende del tiempo; this->get_time() queda disponible.
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


  // ==========================================================
  // Campo Vectorial de velocidad beta
  // ==========================================================
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
    // --- Malla y sistema ---
    void make_grid();
    void setup_system();

    // --- Ensamble (mesh_loop) ---
    void assemble_advection_matrix();                    // A: una vez por malla
    void assemble_boundary_rhs(const double t,
                               Vector<double> &rhs) const; // F(t): inflow débil

    // --- Tiempo ---
    double compute_time_step() const;
    void advance_theta();
    void advance_rk();
    void solve_time_step();
    void apply_inverse_mass(const Vector<double> &src,
                            Vector<double>       &dst) const;

    // --- Adaptación y salida ---
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);
    void output_results(const unsigned int timestep) const;

    // --- Discretización (el orden de declaración es el de inicialización) ---
    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;
    const FE_DGQ<dim>    fe;
    DoFHandler<dim>      dof_handler;

    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> quadrature_face;

    // --- Álgebra lineal ---
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;       // M
    SparseMatrix<double> advection_matrix;  // A
    SparseMatrix<double> system_matrix;     // K = M + k*theta*A (solo theta)

    // Inversa de M por celda (M es diagonal por bloques en DG)
    std::vector<FullMatrix<double>> inverse_mass_cells;

    Vector<double> solution;         // U^n
    Vector<double> old_solution;     // U^{n-1}
    Vector<double> system_rhs;       // F~
    Vector<double> tmp;              // auxiliar (A*U, etc.)
    Vector<double> forcing_terms;    // F(t) del inflow

    // --- Estado temporal ---
    double       time;
    double       time_step;
    unsigned int timestep_number;
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
    , time(t_inicial)
    , time_step(dt_fijo)
    , timestep_number(0)
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

    triangulation.refine_global(refinamiento_inicial);
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

    // Patrón de dispersión DG: incluye el acoplamiento entre celdas
    // vecinas a través de las caras (lo necesita A y, por tanto, K)
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Matrices globales (mismo patrón para M, A y K)
    mass_matrix.reinit(sparsity_pattern);
    advection_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    // Matriz de masa: no hace falta mesh_loop, MatrixCreator basta
    MatrixCreator::create_mass_matrix(mapping,
                                      dof_handler,
                                      quadrature,
                                      mass_matrix);

    // Vectores globales
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    tmp.reinit(dof_handler.n_dofs());
    forcing_terms.reinit(dof_handler.n_dofs());

    // Inversa de M por celda (para RK y θ = 0)
    const unsigned int n = fe.n_dofs_per_cell();
    inverse_mass_cells.assign(triangulation.n_active_cells(),
                              FullMatrix<double>(n, n));
    FullMatrix<double>                   M_local(n, n);
    std::vector<types::global_dof_index> dof_indices(n);

    unsigned int c = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            M_local(i, j) = mass_matrix(dof_indices[i], dof_indices[j]);

        inverse_mass_cells[c].invert(M_local);
        ++c;
      }
  }


  // ==========================================================
  // assemble_advection_matrix()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::assemble_advection_matrix()
  {
    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

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
                                            c.local_dof_indices,
                                            advection_matrix);
      for (const auto &cdf : c.face_data)
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                              cdf.joint_dof_indices,
                                              advection_matrix);
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
    // assemble_boundary_rhs()
    // ==========================================================
    // Ensambla F(t): contribución del inflow débil, -(β·n) g(x,t) ω_i
    // sobre las caras de borde con β·n < 0. Sobrescribe 'rhs'.
    template <int dim>
    void Problema1<dim>::assemble_boundary_rhs(const double   t,
                                              Vector<double> &rhs) const
    {
      using Iterator = typename DoFHandler<dim>::active_cell_iterator;

      BoundaryValues<dim> boundary_function;
      boundary_function.set_time(t);

      rhs = 0;

      // ---------------- Celdas: solo prepara copy_data ----------------
      const auto cell_worker = [&](const Iterator   &cell,
                                  ScratchData<dim> &,
                                  CopyData         &copy_data) {
        copy_data.reinit(cell, fe.n_dofs_per_cell());
      };

      // ---------------- Bordes: solo inflow (β·n < 0) ----------------
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

            if (beta_n < 0) // inflow: -(β·n) g ω_i
              for (unsigned int i = 0; i < n_dofs; ++i)
                copy_data.cell_rhs(i) +=
                  -fe_face.shape_value(i, q) * g[q] * beta_n * JxW[q];
          }
      };

      // ---------------- Copia local -> global: solo el vector ----------------
      const AffineConstraints<double> constraints; // vacío en DG

      const auto copier = [&](const CopyData &c) {
        constraints.distribute_local_to_global(c.cell_rhs,
                                              c.local_dof_indices,
                                              rhs);
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
                              MeshWorker::assemble_boundary_faces,
                            boundary_worker);
    }

    // ==========================================================
    // apply_inverse_mass()
    // ==========================================================
    template <int dim>
    void Problema1<dim>::apply_inverse_mass(const Vector<double> &src,
                                            Vector<double>       &dst) const
    {
      const unsigned int n = fe.n_dofs_per_cell();

      Vector<double>                       src_local(n), dst_local(n);
      std::vector<types::global_dof_index> dof_indices(n);

      unsigned int c = 0;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(dof_indices);

          for (unsigned int i = 0; i < n; ++i)
            src_local(i) = src(dof_indices[i]);

          inverse_mass_cells[c].vmult(dst_local, src_local);   // dst = M_K^{-1} src

          for (unsigned int i = 0; i < n; ++i)
            dst(dof_indices[i]) = dst_local(i);

          ++c;
        }
    }

  // ==========================================================
  // compute_time_step()
  // ==========================================================
  template <int dim>
  double Problema1<dim>::compute_time_step() const
  {
    // Mínimo local de h_K / (|beta|_K (2p+1))  (CFL con factor 1)
    double dt_cfl = std::numeric_limits<double>::max();

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const double h_K = cell->minimum_vertex_distance();

        double beta_max_K = 0.;
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          beta_max_K = std::max(beta_max_K, beta(cell->vertex(v)).norm());

        if (beta_max_K > 1e-14)
          dt_cfl = std::min(dt_cfl, h_K / (beta_max_K * (2. * grado_fe + 1.)));
      }

    if (politica_paso == PoliticaPaso::Fijo)
      {
        const bool explicito =
          (esquema == EsquemaTemporal::RungeKutta) ||
          (esquema == EsquemaTemporal::Theta && theta_valor < 0.5);

        if (explicito && timestep_number == 0 && dt_fijo > dt_cfl)
          std::cerr << "AVISO: dt_fijo = " << dt_fijo
                    << " supera el limite CFL de referencia " << dt_cfl
                    << "; el esquema explicito puede ser inestable."
                    << std::endl;
        return dt_fijo;
      }

    return cfl * dt_cfl;
  }




  // ==========================================================
  // solve_time_step()
  // ==========================================================

  // Se eliminaron las restricciones por nodos flotantes debido a el tipo de elemento a usar

  template <int dim>
  void Problema1<dim>::solve_time_step()
  {
    SolverControl solver_control(gmres_max_iter,
                             gmres_tol_rel * system_rhs.l2_norm());
    SolverGMRES<Vector<double>> solver(solver_control);

    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    std::cout << "   Iterations required for convergence: "
              << solver_control.last_step() << '\n'
              << "   Norm of residual at convergence:    "
              << solver_control.last_value() << '\n';
  }


  // ==========================================================
  // advance_theta()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::advance_theta()
  {
    mass_matrix.vmult(system_rhs, old_solution);
    advection_matrix.vmult(tmp, old_solution);
    system_rhs.add(-(1. - theta_valor) * time_step, tmp);

    assemble_boundary_rhs(time, tmp);
    forcing_terms = tmp;
    forcing_terms *= time_step * theta_valor;

    assemble_boundary_rhs(time - time_step, tmp);
    forcing_terms.add(time_step * (1. - theta_valor), tmp);

    system_rhs += forcing_terms;

    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta_valor * time_step, advection_matrix);

    solve_time_step();
  }

  // ==========================================================
  // advance_rk()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::advance_rk()
  {
    const auto         tabla = TablasRK::obtener_tabla(tabla_rk);
    const unsigned int s     = tabla.etapas;
    const double       t0    = time - time_step;

    std::vector<Vector<double>> K(s, Vector<double>(dof_handler.n_dofs()));
    Vector<double>              Y(dof_handler.n_dofs());

    for (unsigned int i = 0; i < s; ++i)
      {
        Y = old_solution;
        for (unsigned int j = 0; j < i; ++j)
          Y.add(time_step * tabla.A(i, j), K[j]);

        assemble_boundary_rhs(t0 + tabla.c(i) * time_step, forcing_terms);
        advection_matrix.vmult(tmp, Y);
        forcing_terms -= tmp;

        apply_inverse_mass(forcing_terms, K[i]);
      }

    solution = old_solution;
    for (unsigned int i = 0; i < s; ++i)
      solution.add(time_step * tabla.b(i), K[i]);
  }


  // ==========================================================
  // Refine_grid()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::refine_mesh(const unsigned int min_grid_level,
                                   const unsigned int max_grid_level)
  {
    // --- Tu indicador, sin cambios ---
    Vector<float> gradient_indicator(triangulation.n_active_cells());

    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof_handler,
                                                  solution,
                                                  gradient_indicator);

    unsigned int cell_no = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      gradient_indicator(cell_no++) *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    gradient_indicator,
                                                    frac_refinar,
                                                    frac_coarsen);

    // --- Nuevo: respetar niveles mínimo y máximo ---
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();

    // --- Nuevo: transferir la solución entre mallas ---
    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution = solution;

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    triangulation.execute_coarsening_and_refinement();   // tu línea

    // --- Nuevo: reconstruir el sistema en la malla nueva ---
    setup_system();
    assemble_advection_matrix();

    solution_trans.interpolate(solution);
  }



  // ==========================================================
  // output_results()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::output_results(const unsigned int timestep) const
  {
    const std::string carpeta = carpeta_base + std::to_string(dim) + "d";
    fs::create_directories(carpeta);

    const std::string filename = carpeta + "/" + base +
                                 Utilities::int_to_string(timestep, 4) +
                                 extension;
    std::cout << "  Writing solution to <" << filename << '>' << std::endl;
    std::ofstream output(filename);

    DataOut<dim> data_out;
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep));
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u", DataOut<dim>::type_dof_data);

    data_out.build_patches(mapping, fe.degree);

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
      const double l_infty =
        VectorTools::compute_global_error(triangulation,
                                          values,
                                          VectorTools::Linfty_norm);
      std::cout << "  t = " << time << ",  L-infinity norm: " << l_infty
                << std::endl;
    }
  }

  // ==========================================================
  // run()
  // ==========================================================
  template <int dim>
  void Problema1<dim>::run()
  {
    make_grid();          // incluye refine_global(refinamiento_inicial)
    setup_system();
    assemble_advection_matrix();

    if (esquema == EsquemaTemporal::RungeKutta)
      TablasRK::verificar_tabla(TablasRK::obtener_tabla(tabla_rk));

    unsigned int pre_refinement_step = 0;

  start_time_iteration:
    time            = t_inicial;
    timestep_number = 0;

    // Condición inicial u0 = 0
    VectorTools::interpolate(mapping,
                             dof_handler,
                             Functions::ZeroFunction<dim>(),
                             old_solution);
    solution = old_solution;

    output_results(timestep_number);

    // Ciclo de avance temporal
    while (time < t_final - 1e-12)
      {
        time_step = std::min(compute_time_step(), t_final - time);
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        switch (esquema)
          {
            case EsquemaTemporal::Theta:
              advance_theta();
              break;
            case EsquemaTemporal::RungeKutta:
              advance_rk();
              break;
          }
        if (timestep_number % salida_cada == 0)
          output_results(timestep_number);

        if (usar_amr)
          {
            if ((timestep_number == 1) &&
                (pre_refinement_step < n_pre_refinamientos))
              {
                refine_mesh(nivel_min, nivel_max);
                ++pre_refinement_step;
                std::cout << std::endl;
                goto start_time_iteration;
              }
            else if ((timestep_number > 0) &&
                     (timestep_number % pasos_entre_refin == 0))
              refine_mesh(nivel_min, nivel_max);
          }

        old_solution = solution;
      }
  }
}



// ==========================================================
// MAIN
// ==========================================================
int main()
{
  using namespace dealii;
  try
    {
      MultithreadInfo::set_thread_limit();   // opcional (requiere multithread_info.h)
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



















