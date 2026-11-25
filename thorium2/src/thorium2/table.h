// Thorium 2 — tabulated dependencies (DEPEND cards).
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Version 1 interpolated DEPEND tables by the inverse distance method. That method does not
// reproduce even a linear function between the points and gives "steps". Thorium 2 uses:
//   * one argument  — piecewise linear interpolation, constant beyond the first/last point;
//   * two arguments (temperature and time) — bilinear interpolation on a rectangular grid.
// An argument component is active when its values differ between points; a component that
// is blank or has the same value in every point is not an argument (version 1 files write
// "0." for unused components).
#pragma once

#include "thorium2/top.h"

#include <vector>

namespace th {

class Table {
public:
  Table() = default;
  /// Throws InputError (with the card location) if the table cannot be interpolated reliably.
  explicit Table(const TopDepend& d);

  int id() const { return id_; }
  bool depends_on_temperature() const { return t_axis_ >= 0; }
  bool depends_on_time() const { return time_axis_ >= 0; }
  int value_components() const { return n_values_; }

  /// Value component `component` (0-based) at temperature T and time t.
  double eval(double T, double t, int component = 0) const;

  /// Integral of component 0 over temperature from T0 to T1 at fixed time t (exact for the
  /// piecewise-linear interpolant). Used for the enthalpy of c(T).
  double integrate_temperature(double T0, double T1, double t = 0) const;

  /// Temperatures and times of the table points (for validation of property sums).
  const std::vector<double>& temperature_points() const { return t_points_; }
  const std::vector<double>& time_points() const { return time_points_; }

  const std::string& where() const { return where_; }

private:
  double eval_1d(const std::vector<double>& xs, const std::vector<double>& ys, double x) const;
  double integral_1d(const std::vector<double>& xs, const std::vector<double>& ys, double a, double b) const;

  int id_ = 0;
  std::string where_;
  int n_values_ = 0;
  int t_axis_ = -1;     // 0 or 1 position in the grid, -1 if not an argument
  int time_axis_ = -1;
  std::vector<double> t_points_;     // sorted distinct temperatures (empty if not an argument)
  std::vector<double> time_points_;  // sorted distinct times
  // values_[component][i_time * nT + i_T]
  std::vector<std::vector<double>> values_;
};

}  // namespace th
