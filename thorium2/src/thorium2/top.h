// Thorium 2 — thermal-optical properties, loads and solution parameters (*.top file).
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Format: T.H.O.R.I.U.M. 1 user manual, chapter 4. Fields are NASTRAN small fields
// (8 columns). A numeric field is a constant if written as a real ("300.", "1E3") and the
// number of a DEPEND card if written as an integer ("12").
#pragma once

#include "thorium2/cards.h"

#include <array>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace th {

/// A property value: constant or reference to a DEPEND card.
struct TopValue {
  bool is_dependency = false;
  double constant = 0;
  int dependency = 0;

  static TopValue constant_of(double v) { return {false, v, 0}; }
  static TopValue dependency_of(int id) { return {true, 0, id}; }
};

struct TopFtop {
  int id = 0;
  TopValue r_dif, r_ref, absorb, transmit, eps;
  bool eps_given = false;  ///< If not given, emissivity = absorptivity (manual 4.2).
  std::string where;
};

struct TopMedium {
  int id = 0;
  TopValue n, k, lambda, c;
  std::array<int, 6> ftop{};  ///< FTOP numbers of element faces 1..6 (0 = not given).
  std::string where;
};

/// Argument components of a DEPEND point (manual table 4.1).
enum DependArg { kArgTemperature = 0, kArgTime = 1, kArgRandom = 2, kArgFrequency = 3, kArgElevation = 4, kArgAzimuth = 5 };

struct TopDepend {
  int id = 0;
  double idw_power = 2;  ///< "n" of the card: exponent of the inverse distance method of version 1 (not used).
  std::vector<std::array<std::optional<double>, 6>> args;  ///< One entry per point.
  std::vector<std::vector<std::optional<double>>> values;  ///< Up to 8 components per point.
  std::string where;
};

struct TopRsource {
  int id = 0;
  bool position_is_dependency = false;
  int position_dependency = 0;
  TopValue x, y, z;
  TopValue power;        ///< W, emitted uniformly in all directions.
  TopValue temperature;  ///< K; not used for gray sources.
  int n_photons = 0;     ///< 0 = take from I_PARAM.
  std::string where;
};

struct TopHsource {
  int id = 0;
  TopValue power;  ///< W, split equally between the elements.
  std::vector<int> elements;
  std::string where;
};

struct TopTinit {
  int id = 0;
  std::vector<std::pair<int, double>> temperatures;
  std::string where;
};

struct TopTset {
  int id = 0;
  std::vector<std::pair<int, TopValue>> temperatures;
  std::string where;
};

struct TopIparam {
  std::string type;  ///< EULER, EULER_V, ADAMS_4 (version 1) or BDF2, BE.
  double t_start = 0;
  double t_length = 0;
  double t_step = 0;
  int n_photons = 0;
  std::optional<double> kt_min, kt_max;
  std::string smooth;
  int smooth_steps = 0;
  std::string where;
};

struct TopModel {
  std::map<int, TopFtop> ftop;
  std::map<int, TopMedium> media;
  std::map<int, TopDepend> depend;
  std::map<int, TopRsource> rsources;
  std::map<int, TopHsource> hsources;
  std::vector<TopTinit> t_init;
  std::vector<TopTset> t_set;
  std::optional<double> t_all;
  std::optional<int> g_medium;
  std::optional<TopIparam> iparam;
  std::vector<std::string> warnings;
};

/// Builds the *.top model. Unknown cards are errors: a misspelled card must not be dropped.
TopModel parse_top(const std::vector<Card>& cards);
TopModel read_top_file(const std::string& path);

}  // namespace th
