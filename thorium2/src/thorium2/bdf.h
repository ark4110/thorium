// Thorium 2 — finite element model from a NASTRAN Bulk Data File.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "thorium2/cards.h"
#include "thorium2/vec.h"

#include <map>
#include <string>
#include <vector>

namespace th {

enum class ElementType { CTRIA3, CQUAD4, CTETRA, CPENTA, CHEXA };

const char* to_string(ElementType t);
/// 2 for shells (CTRIA3, CQUAD4), 3 for solids.
int dimension(ElementType t);
/// Number of corner grids.
int corner_count(ElementType t);

struct BdfGrid {
  int id = 0;
  Vec3 x;
  int line = 0;
};

struct BdfElement {
  int id = 0;
  ElementType type = ElementType::CTRIA3;
  int pid = 0;
  std::vector<int> grids;  ///< Corner grids only (midside grids of higher-order elements are dropped).
  std::string where;       ///< "file:line" for messages.
};

struct BdfPshell {
  int pid = 0;
  int mid = 0;
  double thickness = 0;  ///< m
  double nsm = 0;        ///< Non-structural mass, kg/m^2.
  std::string where;
};

struct BdfPsolid {
  int pid = 0;
  int mid = 0;
  std::string where;
};

struct BdfMaterial {
  int mid = 0;
  double rho = 0;  ///< kg/m^3
  std::string where;
};

struct BdfModel {
  std::map<int, BdfGrid> grids;
  std::vector<BdfElement> elements;  ///< In file order.
  std::map<int, BdfPshell> pshell;
  std::map<int, BdfPsolid> psolid;
  std::map<int, BdfMaterial> materials;   ///< MAT1 and MAT4 (only density is used).
  std::map<std::string, int> ignored;     ///< Card name -> count of cards that do not affect a thermal analysis.
  std::vector<std::string> warnings;
};

/// Builds a model from cards. Throws InputError on anything that would make the analysis
/// wrong (unsupported coordinate systems, duplicate ids, missing grids, PCOMP, ...).
BdfModel parse_bdf(const std::vector<Card>& cards);
BdfModel read_bdf_file(const std::string& path);

}  // namespace th
