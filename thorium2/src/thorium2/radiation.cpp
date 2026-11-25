// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/radiation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace th {

// ---------------------------------------------------------------- random numbers

namespace {
std::uint64_t splitmix(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
}  // namespace

Random::Random(std::uint64_t seed) {
  for (auto& s : s_) s = splitmix(seed);
}

std::uint64_t Random::mix(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  std::uint64_t x = a;
  std::uint64_t h = splitmix(x);
  x = h ^ b;
  h = splitmix(x);
  x = h ^ c;
  return splitmix(x);
}

double Random::uniform() {
  const std::uint64_t result = s_[0] + s_[3];
  const std::uint64_t t = s_[1] << 17;
  s_[2] ^= s_[0];
  s_[3] ^= s_[1];
  s_[1] ^= s_[2];
  s_[0] ^= s_[3];
  s_[2] ^= t;
  s_[3] = rotl(s_[3], 45);
  return static_cast<double>(result >> 11) * 0x1.0p-53;
}

// ---------------------------------------------------------------- scene

namespace {

struct Frame {
  Vec3 t, b, n;
};

Frame frame(const Vec3& n) {
  const Vec3 helper = std::abs(n.x) < 0.6 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
  const Vec3 t = normalized(cross(helper, n));
  return {t, cross(n, t), n};
}

/// Lambert (cosine) distribution around n.
Vec3 cosine_direction(const Vec3& n, double u1, double u2) {
  const Frame f = frame(n);
  const double r = std::sqrt(u1), phi = 2 * M_PI * u2;
  return normalized(r * std::cos(phi) * f.t + r * std::sin(phi) * f.b + std::sqrt(std::max(0.0, 1 - u1)) * f.n);
}

std::vector<Vec3> face_polygon(const Model& m, const Face& f) {
  std::vector<Vec3> p;
  for (int n : f.nodes) p.push_back(m.nodes[static_cast<size_t>(n)]);
  return p;
}

}  // namespace

RadiationScene::RadiationScene(const Model& model) : model_(model) {
  face_tris_.resize(model.faces.size());
  face_cdf_.resize(model.faces.size());
  std::vector<Bvh::Tri> tris;
  for (size_t fi = 0; fi < model.faces.size(); ++fi) {
    const Face& f = model.faces[fi];
    if (!f.radiating) continue;
    emitters_.push_back(static_cast<int>(fi));
    face_tris_[fi] = triangulate(face_polygon(model, f));
    double acc = 0;
    for (const Triangle& t : face_tris_[fi]) {
      acc += 0.5 * norm(cross(t.b - t.a, t.c - t.a));
      face_cdf_[fi].push_back(acc);
    }
  }
  // Geometry for ray casting: outer faces of solids; one surface per shell (its front face).
  for (size_t ei = 0; ei < model.elements.size(); ++ei) {
    const Element& e = model.elements[ei];
    if (e.dim == 2) {
      const Face& front = model.faces[static_cast<size_t>(e.faces[0])];
      const Face& back = model.faces[static_cast<size_t>(e.faces[1])];
      if (!front.radiating && !back.radiating) continue;
      for (const Triangle& t : triangulate(face_polygon(model, front)))
        tris.push_back({t.a, t.b, t.c, e.faces[0], static_cast<int>(ei)});
    } else {
      for (int fi : e.faces) {
        const Face& f = model.faces[static_cast<size_t>(fi)];
        if (!f.radiating) continue;
        for (const Triangle& t : face_tris_[static_cast<size_t>(fi)]) tris.push_back({t.a, t.b, t.c, fi, static_cast<int>(ei)});
      }
    }
  }
  bvh_ = Bvh(std::move(tris));
  double r = 0;
  model.bounding_sphere(center_, r);
  radius_ = r * 1.0001 + 1e-12;
  eps_ = 1e-9 * std::max(r, 1e-6);
}

Vec3 RadiationScene::sample_point(int face, double u1, double u2, double u3) const {
  const auto& cdf = face_cdf_[static_cast<size_t>(face)];
  const auto& tris = face_tris_[static_cast<size_t>(face)];
  const double x = u1 * cdf.back();
  size_t i = static_cast<size_t>(std::upper_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
  if (i >= tris.size()) i = tris.size() - 1;
  const Triangle& t = tris[i];
  const double s = std::sqrt(u2);
  return (1 - s) * t.a + s * (1 - u3) * t.b + s * u3 * t.c;
}

// ---------------------------------------------------------------- tracing

namespace {

struct FaceOptics {
  double absorb = 1, r_dif = 0, r_ref = 0, transmit = 0;
};

constexpr int kEscaped = -1;
constexpr int kBounceLimit = -2;

class Tracer {
public:
  Tracer(const Model& m, const RadiationScene& s, const std::vector<FaceOptics>& optics, int max_interactions)
      : m_(m), s_(s), optics_(optics), max_(max_interactions) {}

  /// Follows one bundle; returns the absorbing element, kEscaped or kBounceLimit.
  int trace(Vec3 o, Vec3 d, int skip, Random& rng) const {
    constexpr double inf = std::numeric_limits<double>::infinity();
    for (int k = 0; k < max_; ++k) {
      const Bvh::Hit h = s_.bvh().intersect(o, d, inf, skip);
      if (h.tri < 0) return kEscaped;
      const Bvh::Tri& tri = s_.bvh().tri(h.tri);
      const int e = tri.element;
      const Element& el = m_.elements[static_cast<size_t>(e)];
      int side = tri.face;
      if (el.dim == 2 && dot(d, m_.faces[static_cast<size_t>(tri.face)].normal) > 0) side = el.faces[1];
      const Face& f = m_.faces[static_cast<size_t>(side)];
      if (!f.radiating) return e;  // covered side (cannot be reached geometrically)
      const Vec3 n = f.normal;     // points back towards the incoming bundle
      const Vec3 p = o + h.t * d;
      const FaceOptics& op = optics_[static_cast<size_t>(side)];
      const double u = rng.uniform();
      if (u < op.absorb) return e;
      if (u < op.absorb + op.r_dif) {
        d = cosine_direction(n, rng.uniform(), rng.uniform());
        o = p + s_.epsilon() * n;
      } else if (u < op.absorb + op.r_dif + op.r_ref) {
        d = normalized(d - 2 * dot(d, n) * n);
        o = p + s_.epsilon() * n;
      } else if (u < op.absorb + op.r_dif + op.r_ref + op.transmit) {
        o = p - s_.epsilon() * n;
      } else {
        return e;  // rounding of a sum that is 1 within 1e-3: keep the energy in the element
      }
      skip = e;
    }
    return kBounceLimit;
  }

private:
  const Model& m_;
  const RadiationScene& s_;
  const std::vector<FaceOptics>& optics_;
  int max_;
};

}  // namespace

RadiationResult sample_radiation(const Model& m, const RadiationScene& scene, const std::vector<double>& T, double t,
                                 std::uint64_t step, const RadiationSettings& st) {
  RadiationResult r;
  r.absorbed.resize(m.faces.size());
  r.escaped.assign(m.faces.size(), 0.0);
  r.external.assign(m.elements.size(), 0.0);

  // Interaction probabilities of every radiating face at the current temperatures.
  std::vector<FaceOptics> optics(m.faces.size());
  for (size_t fi = 0; fi < m.faces.size(); ++fi) {
    const Face& f = m.faces[fi];
    if (!f.radiating || f.optics < 0) continue;
    const Optics& o = m.optics[static_cast<size_t>(f.optics)];
    const double Te = T[static_cast<size_t>(f.element)];
    optics[fi] = {o.absorb(Te, t), o.r_dif(Te, t), o.r_ref(Te, t), o.transmit(Te, t)};
  }
  const Tracer tracer(m, scene, optics, st.max_interactions);

  long long bounce = 0;
  // ---- emission of faces
  if (st.emit_faces) {
    std::vector<int> faces = st.only_face >= 0 ? std::vector<int>{st.only_face} : scene.emitters();
    const int nf = static_cast<int>(faces.size());
    const int N = st.photons_per_face;
    std::vector<long long> face_bounce(faces.size(), 0);
    [[maybe_unused]] const bool parallel_faces = static_cast<long long>(nf) * N >= 20000;  // threads cost more than tiny jobs
#pragma omp parallel for schedule(dynamic, 1) if (parallel_faces)
    for (int i = 0; i < nf; ++i) {
      const int fi = faces[static_cast<size_t>(i)];
      const Face& f = m.faces[static_cast<size_t>(fi)];
      Random rng(Random::mix(st.seed, step, static_cast<std::uint64_t>(fi)));
      std::vector<int> hits;
      hits.reserve(static_cast<size_t>(N));
      long long escaped = 0;
      for (int k = 0; k < N; ++k) {
        const double u1 = rng.uniform(), u2 = rng.uniform(), u3 = rng.uniform();
        const Vec3 o = scene.sample_point(fi, u1, u2, u3) + scene.epsilon() * f.normal;
        const Vec3 d = cosine_direction(f.normal, rng.uniform(), rng.uniform());
        const int e = tracer.trace(o, d, f.element, rng);
        if (e >= 0) hits.push_back(e);
        else if (e == kEscaped) ++escaped;
        else ++face_bounce[static_cast<size_t>(i)];
      }
      std::sort(hits.begin(), hits.end());
      auto& out = r.absorbed[static_cast<size_t>(fi)];
      for (size_t a = 0; a < hits.size();) {
        size_t b = a;
        while (b < hits.size() && hits[b] == hits[a]) ++b;
        out.push_back({hits[a], static_cast<double>(b - a) / N});
        a = b;
      }
      r.escaped[static_cast<size_t>(fi)] = static_cast<double>(escaped) / N;
    }
    for (long long b : face_bounce) bounce += b;
  }

  // ---- external point sources
  if (st.emit_sources) {
    for (const RadiationSource& src : m.sources) {
      const double Q = src.power(0, t);
      if (!(Q > 0) || src.n_photons <= 0) continue;
      const Vec3 P = src.position_at(t);
      const Vec3 C = scene.center();
      const double R = scene.radius();
      Vec3 w = C - P;
      const double D = norm(w);
      const bool outside = D > R;
      double one_minus_cos = 2.0;  // whole sphere
      if (outside) {
        w = w / D;
        const double ratio = R / D;
        one_minus_cos = ratio * ratio / (1 + std::sqrt(1 - ratio * ratio));  // stable for far sources
      }
      const double aimed = Q * one_minus_cos / 2;
      const int N = src.n_photons;
      const double per_ray = aimed / N;
      const Frame fr = outside ? frame(w) : Frame{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      constexpr int kChunk = 8192;
      const int chunks = (N + kChunk - 1) / kChunk;
      std::vector<std::vector<int>> chunk_hits(static_cast<size_t>(chunks));
      std::vector<long long> chunk_escaped(static_cast<size_t>(chunks), 0), chunk_bounce(static_cast<size_t>(chunks), 0);
#pragma omp parallel for schedule(dynamic, 1) if (chunks > 1)
      for (int c = 0; c < chunks; ++c) {
        Random rng(Random::mix(st.seed ^ 0x5EED5EED5EED5EEDULL, step, (static_cast<std::uint64_t>(src.id) << 32) | static_cast<std::uint64_t>(c)));
        const int n = std::min(kChunk, N - c * kChunk);
        for (int k = 0; k < n; ++k) {
          Vec3 o, d;
          const double phi = 2 * M_PI * rng.uniform();
          if (outside) {
            const double s = rng.uniform() * one_minus_cos;  // 1 - cos(theta)
            const double sin_t = std::sqrt(s * (2 - s));
            const Vec3 u = std::cos(phi) * fr.t + std::sin(phi) * fr.b;
            d = normalized((1 - s) * w + sin_t * u);
            // Start on the near side of the bounding sphere: P + d (D - R), written without
            // cancellation for very distant sources.
            o = C + (-R - (D - R) * s) * w + (D - R) * sin_t * u;
          } else {
            const double z = 1 - 2 * rng.uniform(), rr = std::sqrt(std::max(0.0, 1 - z * z));
            d = {rr * std::cos(phi), rr * std::sin(phi), z};
            o = P;
          }
          const int e = tracer.trace(o, d, -1, rng);
          if (e >= 0) chunk_hits[static_cast<size_t>(c)].push_back(e);
          else if (e == kEscaped) ++chunk_escaped[static_cast<size_t>(c)];
          else ++chunk_bounce[static_cast<size_t>(c)];
        }
      }
      std::vector<long long> count(m.elements.size(), 0);
      long long escaped = 0;
      for (int c = 0; c < chunks; ++c) {
        for (int e : chunk_hits[static_cast<size_t>(c)]) ++count[static_cast<size_t>(e)];
        escaped += chunk_escaped[static_cast<size_t>(c)];
        bounce += chunk_bounce[static_cast<size_t>(c)];
      }
      for (size_t e = 0; e < count.size(); ++e) r.external[e] += static_cast<double>(count[e]) * per_ray;
      r.source_power_aimed += aimed;
      r.source_power_escaped += static_cast<double>(escaped) * per_ray;
    }
  }
  r.bounce_limit_hits = bounce;
  return r;
}

}  // namespace th
