// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/bdf.h"

#include <algorithm>
#include <set>

namespace th {

const char* to_string(ElementType t) {
  switch (t) {
    case ElementType::CTRIA3: return "CTRIA3";
    case ElementType::CQUAD4: return "CQUAD4";
    case ElementType::CTETRA: return "CTETRA";
    case ElementType::CPENTA: return "CPENTA";
    case ElementType::CHEXA: return "CHEXA";
  }
  return "?";
}

int dimension(ElementType t) {
  return (t == ElementType::CTRIA3 || t == ElementType::CQUAD4) ? 2 : 3;
}

int corner_count(ElementType t) {
  switch (t) {
    case ElementType::CTRIA3: return 3;
    case ElementType::CQUAD4: return 4;
    case ElementType::CTETRA: return 4;
    case ElementType::CPENTA: return 6;
    case ElementType::CHEXA: return 8;
  }
  return 0;
}

namespace {

/// Cards that change the thermal model but are not supported: refusing them is better than
/// silently computing a different structure.
const std::set<std::string> kUnsupported = {
    "CBAR",   "CBEAM",  "CROD",   "CONROD", "CTUBE",  "CSHEAR", "CQUAD",  "CQUADR", "CTRIAR",
    "CTRIAX", "CTRIAX6", "CQUADX", "CPYRAM", "CELAS1", "CELAS2", "CELAS3", "CELAS4", "CBUSH",
    "CGAP",   "RBE2",   "RBE3",   "RBAR",   "PCOMP",  "PCOMPG", "CORD1R", "CORD1C", "CORD1S",
    "CORD2R", "CORD2C", "CORD2S", "CORD3G", "CHBDYE", "CHBDYG", "CHBDYP", "PHBDY",  "RADBC",
    "CONV",   "QBDY1",  "QBDY2",  "QVOL",   "QHBDY",  "MAT2",   "MAT8",   "MAT9",   "MAT5",
    "INCLUDE"};

struct ElementCardInfo {
  ElementType type;
  int corners;
  int max_grids;
};

bool element_info(const std::string& name, ElementCardInfo& info) {
  if (name == "CTRIA3") info = {ElementType::CTRIA3, 3, 3};
  else if (name == "CTRIA6") info = {ElementType::CTRIA3, 3, 6};
  else if (name == "CQUAD4") info = {ElementType::CQUAD4, 4, 4};
  else if (name == "CQUAD8") info = {ElementType::CQUAD4, 4, 8};
  else if (name == "CTETRA") info = {ElementType::CTETRA, 4, 10};
  else if (name == "CPENTA") info = {ElementType::CPENTA, 6, 15};
  else if (name == "CHEXA") info = {ElementType::CHEXA, 8, 20};
  else return false;
  return true;
}

std::string at(const Card& c) { return c.file + ":" + std::to_string(c.line); }

void parse_grid(const Card& c, BdfModel& m) {
  BdfGrid g;
  g.id = c.integer(2);
  if (g.id <= 0) c.fail("grid id must be positive");
  const int cp = c.integer(3, 0);
  if (cp != 0)
    c.fail("grid " + std::to_string(g.id) + " uses coordinate system " + std::to_string(cp) +
           "; only the basic system (CP blank or 0) is supported");
  g.x = {c.real(4, 0.0), c.real(5, 0.0), c.real(6, 0.0)};
  g.line = c.line;
  if (!m.grids.emplace(g.id, g).second) c.fail("duplicate grid id " + std::to_string(g.id));
}

void parse_element(const Card& c, const ElementCardInfo& info, BdfModel& m, std::set<int>& ids) {
  BdfElement e;
  e.id = c.integer(2);
  if (e.id <= 0) c.fail("element id must be positive");
  e.type = info.type;
  e.pid = c.integer(3, e.id);  // NASTRAN default: PID = EID
  e.where = at(c);
  for (int k = 0; k < info.corners; ++k) {
    const int idx = 4 + k;
    if (c.blank(idx)) c.fail("element " + std::to_string(e.id) + ": corner grid " + std::to_string(k + 1) + " is blank");
    const int g = c.integer(idx);
    if (g <= 0) c.fail("element " + std::to_string(e.id) + ": grid ids must be positive");
    e.grids.push_back(g);
  }
  std::vector<int> sorted = e.grids;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    c.fail("element " + std::to_string(e.id) + " uses the same grid twice");
  bool midside = false;
  for (int k = info.corners; k < info.max_grids; ++k)
    if (!c.blank(4 + k)) midside = true;
  if (midside)
    m.warnings.push_back(e.where + ": " + c.name + " " + std::to_string(e.id) +
                         ": midside grids are ignored, edges are taken as straight lines");
  if (!ids.insert(e.id).second) c.fail("duplicate element id " + std::to_string(e.id));
  m.elements.push_back(std::move(e));
}

void parse_pshell(const Card& c, BdfModel& m) {
  BdfPshell p;
  p.pid = c.integer(2);
  p.mid = c.integer(3);
  p.thickness = c.real(4);
  if (!(p.thickness > 0)) c.fail("PSHELL " + std::to_string(p.pid) + ": thickness must be positive");
  p.nsm = c.real(9, 0.0);
  if (p.nsm < 0) c.fail("PSHELL " + std::to_string(p.pid) + ": NSM must not be negative");
  p.where = at(c);
  if (m.pshell.count(p.pid) || m.psolid.count(p.pid)) c.fail("duplicate property id " + std::to_string(p.pid));
  m.pshell.emplace(p.pid, p);
}

void parse_psolid(const Card& c, BdfModel& m) {
  BdfPsolid p;
  p.pid = c.integer(2);
  p.mid = c.integer(3);
  p.where = at(c);
  if (m.pshell.count(p.pid) || m.psolid.count(p.pid)) c.fail("duplicate property id " + std::to_string(p.pid));
  m.psolid.emplace(p.pid, p);
}

void parse_material(const Card& c, int rho_field, BdfModel& m) {
  BdfMaterial mat;
  mat.mid = c.integer(2);
  mat.rho = c.real(rho_field, 0.0);
  if (mat.rho < 0) c.fail("material " + std::to_string(mat.mid) + ": density must not be negative");
  mat.where = at(c);
  if (!m.materials.emplace(mat.mid, mat).second) c.fail("duplicate material id " + std::to_string(mat.mid));
}

}  // namespace

BdfModel parse_bdf(const std::vector<Card>& cards) {
  BdfModel m;
  std::set<int> element_ids;
  for (const Card& c : cards) {
    ElementCardInfo info{};
    if (c.name == "GRID") parse_grid(c, m);
    else if (element_info(c.name, info)) parse_element(c, info, m, element_ids);
    else if (c.name == "PSHELL") parse_pshell(c, m);
    else if (c.name == "PSOLID") parse_psolid(c, m);
    else if (c.name == "MAT1") parse_material(c, 6, m);
    else if (c.name == "MAT4") parse_material(c, 5, m);
    else if (kUnsupported.count(c.name))
      c.fail("card is not supported by Thorium 2 (it would change the thermal model); remove it or convert the structure");
    else ++m.ignored[c.name];
  }
  return m;
}

BdfModel read_bdf_file(const std::string& path) { return parse_bdf(read_cards_file(path)); }

}  // namespace th
