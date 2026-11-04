/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 1999 - 2023 by the deal.II authors
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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

void first_grid()
{
  Triangulation<2> rect1;
  Triangulation<2> rect2;
  Triangulation<2> solucion;
  
  // Rectángulo SUPERIOR: de y=0.5 a y=1.0
  // Usamos {1, 1} para tener una sola celda grande
  GridGenerator::subdivided_hyper_rectangle(rect1, {2, 1}, Point<2>(0, 0.5), Point<2>(1, 1.0));

  // Rectángulo INFERIOR: de y=0.0 a y=0.5
  // ¡Fíjate que el techo de este (0.5) coincide con el piso del otro!
  GridGenerator::subdivided_hyper_rectangle(rect2, {1, 1}, Point<2>(0.5, 0), Point<2>(1, 0.5));

  // Unimos las piezas
  GridGenerator::merge_triangulations(rect1, rect2, solucion);

  solucion.refine_global(5);

  std::ofstream out("grid-1.svg");
  GridOut       grid_out;
  grid_out.write_svg(solucion, out);
  std::cout << "Grid written to grid-1.svg" << std::endl;
}

int main()
{
  first_grid();
}