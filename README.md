# FEM for Compressible Flow — deal.II

Hi! 👋 This is the deal.II repository of **SIFLUX-unal**, an undergraduate research group at Universidad Nacional de Colombia working on numerical tools for compressible flow simulation.

<p align="center">
  <img src="images/P04-placa-L.png" width="500" alt="Convection–diffusion solution on an L-shaped plate">
</p>

## What's in here?

A collection of finite element solvers written in [deal.II](https://www.dealii.org/) (C++). We're learning by doing: we start from the official deal.II tutorials, rebuild them block by block, and adapt them to our own problems. Each solver adds something new on top of the previous one, all the way up to compressible flow.

## Roadmap

Where we are and where we're heading:

| Stage | Topic | Status |
|:-----:|-------|:------:|
| 01 | Steady heat conduction | ✅ |
| 02 | Transient heat conduction | ✅ |
| 03 | Adaptive mesh refinement | ✅ |
| 04 | Convection–diffusion with FEM + SUPG | ✅ |
| 05 | Convection–diffusion with DGFEM | ✅ |
| 06 | Nonlinear diffusion–advection–reaction | 🚧 |
| 07 | Euler equations | ⏳ |
| 08 | Compressible Navier–Stokes | ⏳ |

## Problems so far

| ID | What it solves | How |
|----|----------------|-----|
| P04 | Convection–diffusion on an L-shaped plate | FEM + SUPG, parallel assembly with `WorkStream` (based on step-9) |
| P05 | Steady convection–diffusion | Discontinuous Galerkin (based on step-12) |

## How it's organized

Every problem has its own folder with its source file and `CMakeLists.txt`, grouped by roadmap stage:

```
.
├── 04-adveccion-fem/
│   └── P04-placa-L-supg/
├── 05-adveccion-dgfem/
│   └── P05-conv-diff-dg/
├── images/
└── README.md
```

## What you need

- deal.II 9.7.1
- CMake and a C++17 compiler
- ParaView or VisIt to look at the results

## Running a solver

Go into any problem folder and:

```bash
cmake -S . -B build
cmake --build build
./build/<executable_name>
```

Tip: use `cmake --build build --target release` for faster runs, and `--target debug` while you're developing.

## Want to contribute?

If you're part of the group, go for it! Just:

1. Make a branch for your task
2. Commit with clear messages (e.g. `P05: add penalty term`)
3. Open a Pull Request and ask someone to review it

And please don't upload build folders or simulation results 🙏

## License

LGPL-2.1-or-later, same as deal.II. Some solvers are adapted from the [deal.II tutorials](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html), so huge thanks to the deal.II team!
