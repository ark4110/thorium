// Thorium 2 — thermal model: one finite element = one thermal node.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "thorium2/bdf.h"
#include "thorium2/geometry.h"
#include "thorium2/table.h"
#include "thorium2/top.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace th {

/// A material or optical property: constant or table of temperature and/or time.
struct Property {
  double constant = 0;
  const Table* table = nullptr;
  int component = 0;

  double operator()(double T, double t) const { return table ? table->eval(T, t, component) : constant; }
  bool depends_on_temperature() const { return table && table->depends_on_temperature(); }
  bool depends_on_time() const { return table && table->depends_on_time(); }
};

/// Thermal-optical properties of one side of a face (FTOP). The four interaction
/// probabilities sum to 1; checked when the model is built.
struct Optics {
  int id = 0;
  Property r_dif, r_ref, absorb, transmit, eps;
};

struct Medium {
  int id = 0;
  Property lambda;  ///< W/(m K)
  Property c;       ///< J/(kg K); may depend on temperature only.

  /// Specific enthalpy relative to 0 K, J/kg: integral of c from 0 to T.
  double enthalpy(double T) const {
    return c.table ? c.table->integrate_temperature(0.0, T) : c.constant * T;
  }
};

struct Face {
  int element = -1;
  int local = 0;              ///< Face number of the element (manual table 4.2), 1-based.
  std::vector<int> nodes;     ///< Node indices, ordered so that the normal points outward.
  Vec3 normal;                ///< Unit normal pointing away from the material.
  Vec3 centroid;
  double area = 0;
  int optics = -1;            ///< Index into Model::optics; -1 only for non-radiating faces.
  bool radiating = true;      ///< False for faces in contact with another element.
};

struct Element {
  int id = 0;
  ElementType type{};
  int dim = 2;
  int pid = 0;
  int medium = -1;            ///< Index into Model::media.
  std::vector<int> nodes;     ///< Node indices of the corners.
  Vec3 centroid;
  double volume = 0;          ///< m^3 (shells: area * thickness)
  double area = 0;            ///< Shells: mid-surface area. Solids: 0.
  double thickness = 0;       ///< Shells only.
  double mass = 0;            ///< kg, including non-structural mass.
  std::vector<int> faces;     ///< Indices into Model::faces.
  std::string where;
};

/// Heat conduction between elements that share a face (3D), an edge (shells) or a shell
/// lying on a solid face. A junction with k members is the exact elimination of a common
/// node without heat capacity: member a exchanges g_a (T_j - T_a) with
/// T_j = sum(g_b T_b) / sum(g_b), where g_a = lambda_a(T_a) * geometric factor.
/// For two members this is the series resistance A / (d1/lambda1 + d2/lambda2).
struct Junction {
  struct Member {
    int element = -1;
    double geometry = 0;  ///< m: area / distance (3D), thickness * edge length / distance (shell edge).
  };
  std::vector<Member> members;
};

struct RadiationSource {
  int id = 0;
  bool position_is_table = false;
  const Table* position = nullptr;   ///< Value components 1..3 = x, y, z.
  Property x, y, z;
  Property power;                     ///< W, isotropic.
  int n_photons = 0;
  std::string where;

  Vec3 position_at(double t) const {
    if (position_is_table) return {position->eval(0, t, 0), position->eval(0, t, 1), position->eval(0, t, 2)};
    return {x(0, t), y(0, t), z(0, t)};
  }
};

struct HeatSource {
  int id = 0;
  Property power;                     ///< W, split equally over the elements.
  std::vector<int> elements;          ///< Element indices.
};

struct FixedTemperature {
  int element = -1;
  Property temperature;               ///< K, function of time.
};

struct Integration {
  std::string type;                   ///< As written in the file.
  double t_start = 0;
  double t_end = 0;
  double dt = 0;
  int n_photons = 1000;
};

struct Model {
  std::vector<Vec3> nodes;
  std::vector<int> node_ids;          ///< GRID id of each node.
  std::vector<Element> elements;
  std::map<int, int> element_index;   ///< Element id -> index.
  std::vector<Face> faces;
  std::vector<Medium> media;
  std::vector<Optics> optics;
  std::vector<Junction> junctions;
  std::vector<RadiationSource> sources;
  std::vector<HeatSource> heat_sources;
  std::vector<FixedTemperature> fixed;
  std::vector<double> initial_temperature;  ///< Per element.
  Integration integration;
  std::vector<std::string> warnings;

  /// Owns the DEPEND tables referenced by properties (stable addresses).
  std::map<int, std::unique_ptr<Table>> tables;

  int element(int id) const;          ///< Index by id; throws std::out_of_range.
  /// Bounding sphere of all radiating faces.
  void bounding_sphere(Vec3& center, double& radius) const;
};

/// Builds and validates the thermal model. Throws InputError with file:line on any problem.
Model build_model(const BdfModel& bdf, const TopModel& top);
/// Reads <path>.bdf (or .dat) and the *.top file. If `top_path` is empty, the *.top file
/// with the same name is used (as in version 1).
Model load_model(const std::string& bdf_path, const std::string& top_path = {});

}  // namespace th
