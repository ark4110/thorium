// Thorium 2 — writing results.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "thorium2/model.h"
#include "thorium2/solver.h"

#include <fstream>
#include <string>
#include <vector>

namespace th {

/// Text results in the format of T.H.O.R.I.U.M. 1 (Form1.Save_txt_results, option output_txt):
///
///   Element_number Time,s Temperature,K  Input_heat_power,W
///   <element id> <time> <temperature> <net heat power into the element>
///
/// Values are separated by one space, rounded to 6 decimals and written like .NET
/// Double.ToString() (decimal point, up to 15 significant digits, "1E-06" style exponents).
/// Lines are grouped by element: all written steps of the first element, then of the second,
/// in the order of the *.bdf file. The heat power at the initial time is 0, as in version 1.
/// The file is written by save(), after the integration (as version 1 did).
class ResultFile {
public:
  ResultFile(std::string path, const Model& model);
  /// Remembers one time point; `power` may be empty (written as 0).
  void add(double t, const std::vector<double>& temperature, const std::vector<double>& power);
  void save() const;
  const std::string& path() const { return path_; }

private:
  std::string path_;
  std::vector<int> ids_;
  std::vector<double> times_;
  std::vector<std::vector<double>> T_, Q_;  // per time point
};

/// Number as version 1 wrote it: Math.Round(v, 6) then Double.ToString().
std::string v1_number(double v);

/// Power balance of the model and solver statistics per step, space separated text.
class BalanceTable {
public:
  explicit BalanceTable(const std::string& path);
  void write(const StepReport& r);

private:
  std::ofstream out_;
};

/// Geometry and properties of the model for a viewer (JSON): nodes, and per element its
/// visible polygons (shells: the mid-surface; solids: outer faces) as node indices.
void write_mesh_json(const std::string& path, const Model& model);

/// Number of CPUs this process may use (cgroup quota, affinity), at least 1.
int available_cpus();

}  // namespace th
