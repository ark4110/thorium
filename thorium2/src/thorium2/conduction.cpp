// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/conduction.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace th {

namespace {

[[noreturn]] void fail(const Element& e, const std::string& msg) {
  throw InputError(e.where + ": " + to_string(e.type) + " " + std::to_string(e.id) + ": " + msg);
}

/// Distance from the element centroid to the plane of its face, along the face normal.
double normal_distance(const Model& m, const Face& f) {
  const Element& e = m.elements[static_cast<size_t>(f.element)];
  const double d = std::abs(dot(f.centroid - e.centroid, f.normal));
  if (!(d > 1e-9 * std::cbrt(e.volume))) fail(e, "centroid lies on face " + std::to_string(f.local) + " (degenerate element)");
  return d;
}

}  // namespace

void build_junctions(Model& m) {
  // --- Contacts through a whole face: solid-solid, shell-on-solid. ---
  std::map<std::vector<int>, std::vector<int>> solid_faces;  // sorted nodes -> face indices
  std::map<std::vector<int>, std::vector<int>> shells;       // sorted nodes -> element indices
  for (size_t ei = 0; ei < m.elements.size(); ++ei) {
    const Element& e = m.elements[ei];
    if (e.dim == 3) {
      for (int fi : e.faces) {
        std::vector<int> key = m.faces[static_cast<size_t>(fi)].nodes;
        std::sort(key.begin(), key.end());
        solid_faces[key].push_back(fi);
      }
    } else {
      std::vector<int> key = e.nodes;
      std::sort(key.begin(), key.end());
      shells[key].push_back(static_cast<int>(ei));
    }
  }
  for (const auto& [key, list] : shells)
    if (list.size() > 1)
      fail(m.elements[static_cast<size_t>(list[1])],
           "lies on the same grids as element " + std::to_string(m.elements[static_cast<size_t>(list[0])].id) +
               " (coincident shells are not supported; offset them or merge into one element)");

  for (const auto& [key, faces] : solid_faces) {
    if (faces.size() > 2)
      fail(m.elements[static_cast<size_t>(m.faces[static_cast<size_t>(faces[2])].element)],
           "more than two solid elements share one face (the mesh is not manifold)");
    auto sh = shells.find(key);
    if (sh == shells.end()) {
      if (faces.size() == 2) {
        Face& a = m.faces[static_cast<size_t>(faces[0])];
        Face& b = m.faces[static_cast<size_t>(faces[1])];
        const double A = 0.5 * (a.area + b.area);
        m.junctions.push_back({{{a.element, A / normal_distance(m, a)}, {b.element, A / normal_distance(m, b)}}});
        a.radiating = b.radiating = false;
      }
      continue;
    }
    // A shell lying on one solid face, or sandwiched between two solids.
    const int shell = sh->second[0];
    const Element& s = m.elements[static_cast<size_t>(shell)];
    for (int fi : faces) {
      Face& sf = m.faces[static_cast<size_t>(fi)];
      const double A = sf.area;
      m.junctions.push_back({{{sf.element, A / normal_distance(m, sf)}, {shell, A / (0.5 * s.thickness)}}});
      sf.radiating = false;
      // The side of the shell that faces the solid is covered.
      for (int shf : s.faces) {
        Face& f = m.faces[static_cast<size_t>(shf)];
        if (dot(f.normal, sf.normal) < 0) f.radiating = false;
      }
    }
  }

  // --- Shell edges: any number of shells at any angle meeting on one edge. ---
  std::map<std::pair<int, int>, std::vector<int>> edges;  // (min node, max node) -> shell elements
  for (size_t ei = 0; ei < m.elements.size(); ++ei) {
    const Element& e = m.elements[ei];
    if (e.dim != 2) continue;
    const size_t n = e.nodes.size();
    for (size_t k = 0; k < n; ++k) {
      int a = e.nodes[k], b = e.nodes[(k + 1) % n];
      if (a > b) std::swap(a, b);
      edges[{a, b}].push_back(static_cast<int>(ei));
    }
  }
  std::set<int> solid_nodes;
  for (const Element& e : m.elements)
    if (e.dim == 3) solid_nodes.insert(e.nodes.begin(), e.nodes.end());
  int loose_edges = 0;
  for (const auto& [edge, list] : edges) {
    if (list.size() < 2) {
      if (solid_nodes.count(edge.first) && solid_nodes.count(edge.second)) ++loose_edges;
      continue;
    }
    const Vec3& a = m.nodes[static_cast<size_t>(edge.first)];
    const Vec3& b = m.nodes[static_cast<size_t>(edge.second)];
    const double L = norm(b - a);
    Junction j;
    for (int ei : list) {
      const Element& e = m.elements[static_cast<size_t>(ei)];
      const double d = distance_to_line(e.centroid, a, b);
      if (!(d > 1e-9 * std::sqrt(e.area))) fail(e, "centroid lies on an edge (degenerate element)");
      j.members.push_back({ei, e.thickness * L / d});
    }
    m.junctions.push_back(std::move(j));
  }
  if (loose_edges > 0)
    m.warnings.push_back(std::to_string(loose_edges) +
                         " shell edge(s) lie on grids of solid elements without a thermal contact (a shell edge "
                         "touching a solid conducts no heat; put the shell on a solid face to connect it)");
}

std::vector<double> element_conductivity(const Model& m, const std::vector<double>& T, double t) {
  std::vector<double> lambda(m.elements.size());
  for (size_t i = 0; i < m.elements.size(); ++i)
    lambda[i] = m.media[static_cast<size_t>(m.elements[i].medium)].lambda(T[i], t);
  return lambda;
}

void add_conduction(const Model& m, const std::vector<double>& T, const std::vector<double>& lambda, std::vector<double>& q,
                    std::vector<Triplet>* J) {
  std::vector<double> g;
  for (const Junction& j : m.junctions) {
    g.resize(j.members.size());
    double sum = 0, sumT = 0;
    for (size_t k = 0; k < j.members.size(); ++k) {
      const size_t e = static_cast<size_t>(j.members[k].element);
      g[k] = lambda[e] * j.members[k].geometry;
      sum += g[k];
      sumT += g[k] * T[e];
    }
    if (!(sum > 0)) continue;
    const double Tj = sumT / sum;
    for (size_t k = 0; k < j.members.size(); ++k) {
      const int a = j.members[k].element;
      q[static_cast<size_t>(a)] += g[k] * (Tj - T[static_cast<size_t>(a)]);
      if (!J) continue;
      for (size_t l = 0; l < j.members.size(); ++l) {
        const double v = g[k] * g[l] / sum - (k == l ? g[k] : 0.0);
        J->push_back({a, j.members[l].element, v});
      }
    }
  }
}

}  // namespace th
