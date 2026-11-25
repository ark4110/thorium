// Thorium 2 — heat conduction between elements (method of thermal balances, written so that
// it is exact for series resistances and for junctions of several shells).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "thorium2/model.h"

#include <vector>

namespace th {

/// Finds contacts and fills model.junctions; marks contact faces non-radiating.
/// Called by build_model.
void build_junctions(Model& model);

/// Thermal conductivity of every element at temperatures T and time t.
std::vector<double> element_conductivity(const Model& model, const std::vector<double>& T, double t);

/// Adds the conductive heat flow into each element (W) to `q`.
/// If `jacobian` is given, adds d(q_a)/d(T_b) at fixed conductivities as (a, b, value) triplets.
struct Triplet {
  int row, col;
  double value;
};
void add_conduction(const Model& model, const std::vector<double>& T, const std::vector<double>& lambda,
                    std::vector<double>& q, std::vector<Triplet>* jacobian = nullptr);

}  // namespace th
