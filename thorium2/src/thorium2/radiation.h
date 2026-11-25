// Thorium 2 — radiation heat transfer by the Monte Carlo method.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// On every time step each radiating face emits N "photon bundles" from random points, in
// directions with Lambert's (cosine) distribution. A bundle is followed through absorption,
// diffuse and specular reflection and transmission until it is absorbed or leaves the model.
// The result is, for every face f, the fraction F(f -> e) of its emission absorbed by every
// element e. The solver uses these fractions implicitly: the power emitted by f is
// eps_f sigma A_f T^4 at the new temperature, so the T^4 coupling is treated by Newton's
// method and the scheme stays stable for any step.
//
// External sources (RSOURCE1) are isotropic points; their bundles are sent only into the cone
// that contains the model, each carrying the matching share of the source power.
#pragma once

#include "thorium2/bvh.h"
#include "thorium2/model.h"

#include <cstdint>
#include <vector>

namespace th {

constexpr double kStefanBoltzmann = 5.670374419e-8;  // W/(m^2 K^4), CODATA 2018

/// Geometry prepared for ray casting.
class RadiationScene {
public:
  explicit RadiationScene(const Model& model);

  const Bvh& bvh() const { return bvh_; }
  /// Radiating faces (indices into Model::faces) in a fixed order.
  const std::vector<int>& emitters() const { return emitters_; }
  Vec3 center() const { return center_; }
  double radius() const { return radius_; }
  double epsilon() const { return eps_; }

  /// Uniform random point on a face, from two uniform numbers.
  Vec3 sample_point(int face, double u1, double u2, double u3) const;

private:
  const Model& model_;
  Bvh bvh_;
  std::vector<int> emitters_;
  std::vector<std::vector<Triangle>> face_tris_;    // per model face
  std::vector<std::vector<double>> face_cdf_;       // cumulative triangle areas
  Vec3 center_;
  double radius_ = 0;
  double eps_ = 0;
};

struct RadiationResult {
  /// For every model face: (element, fraction of the face's emission absorbed by it).
  /// Empty for non-radiating faces.
  std::vector<std::vector<std::pair<int, double>>> absorbed;
  /// For every model face: fraction of emission that left the model.
  std::vector<double> escaped;
  /// Power from external sources absorbed by each element, W.
  std::vector<double> external;
  double source_power_aimed = 0;   ///< W sent towards the model by all sources.
  double source_power_escaped = 0; ///< W of it that left the model.
  long long bounce_limit_hits = 0; ///< Bundles stopped after too many interactions.
};

struct RadiationSettings {
  int photons_per_face = 1000;
  std::uint64_t seed = 1;
  int max_interactions = 10000;
  /// If >= 0, only this face emits (used for configuration factors in tests).
  int only_face = -1;
  bool emit_faces = true;
  bool emit_sources = true;
};

/// One Monte Carlo sample. Optical properties are taken at temperatures T (per element) and
/// time t. Results do not depend on the number of threads.
RadiationResult sample_radiation(const Model& model, const RadiationScene& scene, const std::vector<double>& T, double t,
                                 std::uint64_t step, const RadiationSettings& settings);

/// Deterministic random numbers (xoshiro256+ seeded through splitmix64).
class Random {
public:
  explicit Random(std::uint64_t seed);
  static std::uint64_t mix(std::uint64_t a, std::uint64_t b, std::uint64_t c);
  double uniform();  ///< [0, 1)

private:
  std::uint64_t s_[4];
};

}  // namespace th
