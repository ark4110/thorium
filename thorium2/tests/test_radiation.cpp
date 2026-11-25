// Radiation by Monte Carlo against configuration factors from Howell's catalog (the same
// formulas the dissertation used, eqs. 3.1-3.2). Tolerances are 4 standard deviations of the
// binomial estimate: sigma = sqrt(F (1 - F) / N).
#include "reference.h"
#include "thorium2/radiation.h"

#include <gtest/gtest.h>

#ifdef THORIUM2_HAVE_OPENMP
#include <omp.h>
#endif

using namespace th;
using ref::card;
using ref::num;

namespace {

const std::string kProps = card("PSHELL", {"10", "1", ".001"}) + card("MAT1", {"1", "", "", "", "1000."});

std::string top(int photons, const std::string& extra = "") {
  return ref::top_header(1.0, 1.0, photons, 300.0) + card("MEDIUM1", {"10", "1.", "0.", "1.", "1000."}) + ",1,1\n" + extra;
}

/// Fraction of the emission of element `from` (front face) absorbed by element `to`.
double fraction(const Model& m, int from, int to, int photons, std::uint64_t seed = 1) {
  RadiationScene scene(m);
  RadiationSettings s;
  s.photons_per_face = photons;
  s.seed = seed;
  s.only_face = m.elements[static_cast<size_t>(from)].faces[0];
  s.emit_sources = false;
  auto r = sample_radiation(m, scene, m.initial_temperature, 0.0, 0, s);
  double f = 0;
  for (auto [e, v] : r.absorbed[static_cast<size_t>(s.only_face)])
    if (e == to) f += v;
  return f;
}

double tol(double F, int n) { return 4.0 * std::sqrt(std::max(F * (1 - F), 1e-6) / n); }

/// Two parallel a x b rectangles at distance c, front faces towards each other.
Model parallel(double a, double b, double c, const std::string& extra_top = "", int photons = 10) {
  std::string bdf = ref::plate(1, 1, a, b, 0.0, 10, 0, 0) + ref::plate(1, 1, a, b, c, 10, 4, 1, true) + kProps;
  return ref::model_from_text(bdf, top(photons, extra_top));
}

/// Unit cube of six CQUAD4 (property 10) with front faces pointing inward.
std::string unit_box() {
  std::string bdf;
  const char* g[8][3] = {{"0.", "0.", "0."}, {"1.", "0.", "0."}, {"1.", "1.", "0."}, {"0.", "1.", "0."},
                         {"0.", "0.", "1."}, {"1.", "0.", "1."}, {"1.", "1.", "1."}, {"0.", "1.", "1."}};
  for (int i = 0; i < 8; ++i) bdf += card("GRID", {std::to_string(i + 1), "", g[i][0], g[i][1], g[i][2]});
  const int q[6][4] = {{1, 2, 3, 4}, {5, 8, 7, 6}, {1, 5, 6, 2}, {2, 6, 7, 3}, {3, 7, 8, 4}, {4, 8, 5, 1}};
  for (int i = 0; i < 6; ++i)
    bdf += card("CQUAD4", {std::to_string(i + 1), "10", std::to_string(q[i][0]), std::to_string(q[i][1]),
                           std::to_string(q[i][2]), std::to_string(q[i][3])});
  return bdf;
}

}  // namespace

TEST(Radiation, ParallelSquaresMatchCatalog) {
  // Fig. 3.6 of the dissertation: unit squares, distance c from 0.05 to 2.
  const int N = 400000;
  for (double c : {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}) {
    Model m = parallel(1.0, 1.0, c);
    const double F = ref::F_parallel_rectangles(1.0, 1.0, c);
    EXPECT_NEAR(fraction(m, 0, 1, N), F, tol(F, N)) << "c = " << c;
  }
  // Rectangles 2 x 0.5 at 0.3.
  Model m = parallel(2.0, 0.5, 0.3);
  const double F = ref::F_parallel_rectangles(2.0, 0.5, 0.3);
  EXPECT_NEAR(fraction(m, 0, 1, N), F, tol(F, N));
}

TEST(Radiation, PerpendicularRectanglesMatchCatalog) {
  // Figs. 3.3-3.5: common edge l, rectangle 1 of width w in z = 0, rectangle 2 of height h in x = 0.
  const int N = 200000;
  for (double w : {0.1, 1.0, 10.0})
    for (double l : {0.1, 1.0, 10.0})
      for (double h : {0.2, 1.0}) {
        std::string bdf = card("GRID", {"1", "", "0.", "0.", "0."}) + card("GRID", {"2", "", num(w), "0.", "0."}) +
                          card("GRID", {"3", "", num(w), num(l), "0."}) + card("GRID", {"4", "", "0.", num(l), "0."}) +
                          card("GRID", {"5", "", "0.", "0.", num(h)}) + card("GRID", {"6", "", "0.", num(l), num(h)}) +
                          card("CQUAD4", {"1", "10", "1", "2", "3", "4"}) +  // normal +z
                          card("CQUAD4", {"2", "10", "1", "4", "6", "5"}) +  // normal +x
                          kProps;
        Model m = ref::model_from_text(bdf, top(10));
        const double F = ref::F_perpendicular_rectangles(h, w, l);
        EXPECT_NEAR(fraction(m, 0, 1, N), F, tol(F, N)) << "w=" << w << " l=" << l << " h=" << h;
      }
}

TEST(Radiation, ReciprocityAndClosedBox) {
  // Unit cube of six black plates with normals inward: all emission is absorbed inside.
  std::string bdf = unit_box();
  bdf += kProps;
  Model m = ref::model_from_text(bdf, top(100000));
  RadiationScene scene(m);
  RadiationSettings s;
  s.photons_per_face = 100000;
  s.emit_sources = false;
  auto r = sample_radiation(m, scene, m.initial_temperature, 0.0, 0, s);
  for (int e = 0; e < 6; ++e) {
    const int inner = m.elements[static_cast<size_t>(e)].faces[0];
    double sum = 0;
    for (auto [el, v] : r.absorbed[static_cast<size_t>(inner)]) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-12) << "closed box leaks";
    EXPECT_EQ(r.escaped[static_cast<size_t>(inner)], 0.0);
    // Outer faces see nothing.
    const int outer = m.elements[static_cast<size_t>(e)].faces[1];
    EXPECT_NEAR(r.escaped[static_cast<size_t>(outer)], 1.0, 1e-12);
  }
  // Opposite faces of a cube: F = 0.19982; adjacent: 0.20004.
  const double Fo = ref::F_parallel_rectangles(1, 1, 1), Fa = ref::F_perpendicular_rectangles(1, 1, 1);
  auto F = [&](int a, int b) {
    double f = 0;
    for (auto [el, v] : r.absorbed[static_cast<size_t>(m.elements[static_cast<size_t>(a)].faces[0])])
      if (el == b) f += v;
    return f;
  };
  EXPECT_NEAR(F(0, 1), Fo, tol(Fo, 100000));
  EXPECT_NEAR(F(0, 2), Fa, tol(Fa, 100000));
  EXPECT_NEAR(F(3, 5), Fo, tol(Fo, 100000));
}

TEST(Radiation, SpecularMirrorDoublesDistance) {
  // Black plate 1 under a perfect specular mirror (plate 2) at distance c: what plate 1 absorbs
  // from itself equals the factor to its image at distance 2c.
  const std::string mirror = card("FTOP", {"2", "0.", "1.", "0.", "0.", "0."}) +
                             card("MEDIUM1", {"11", "1.", "0.", "1.", "1000."}) + ",2,2\n";
  std::string bdf = ref::plate(1, 1, 1.0, 1.0, 0.0, 10, 0, 0) + ref::plate(1, 1, 1.0, 1.0, 0.4, 11, 4, 1, true) + kProps +
                    card("PSHELL", {"11", "1", ".001"});
  Model m = ref::model_from_text(bdf, top(10, mirror));
  const int N = 400000;
  const double F2 = ref::F_parallel_rectangles(1, 1, 0.8);
  EXPECT_NEAR(fraction(m, 0, 0, N), F2, tol(F2, N));
  EXPECT_EQ(fraction(m, 0, 1, N), 0.0);  // the mirror absorbs nothing
}

TEST(Radiation, DiffuseReflectorFirstHit) {
  // Plate 2 reflects diffusely with rho = 0.6 and absorbs 1 - rho: it absorbs exactly
  // (1 - rho) F12 of the emission of plate 1 (the first hit does not depend on reflections).
  const double rho = 0.6;
  const std::string gray = card("FTOP", {"2", num(rho), "0.", num(1 - rho), "0."}) +
                           card("MEDIUM1", {"11", "1.", "0.", "1.", "1000."}) + ",2,2\n";
  std::string bdf = ref::plate(1, 1, 1.0, 1.0, 0.0, 10, 0, 0) + ref::plate(1, 1, 1.0, 1.0, 0.5, 11, 4, 1, true) + kProps +
                    card("PSHELL", {"11", "1", ".001"});
  Model m = ref::model_from_text(bdf, top(10, gray));
  const int N = 400000;
  const double F = ref::F_parallel_rectangles(1, 1, 0.5);
  EXPECT_NEAR(fraction(m, 0, 1, N), (1 - rho) * F, tol((1 - rho) * F, N));
  // Some of the reflected energy comes back to plate 1, never more than rho F.
  const double back = fraction(m, 0, 0, N);
  EXPECT_GT(back, 0.0);
  EXPECT_LT(back, rho * F);
}

TEST(Radiation, GrayDiffuseEnclosureObeysGebhartReciprocity) {
  // Closed unit cube, inner sides gray and diffuse (absorptivity 0.3): every bundle is absorbed
  // inside, and the Gebhart factors obey A_i a_i G_ij = A_j a_j G_ji (equal here).
  std::string bdf = unit_box() + kProps;
  const std::string gray = card("FTOP", {"2", "0.7", "0.", "0.3", "0.", "0.3"}) +
                           ref::top_header(1.0, 1.0, 10, 300.0) + card("MEDIUM1", {"10", "1.", "0.", "1.", "1000."}) + ",2,1\n";
  Model m = ref::model_from_text(bdf, gray);
  const int N = 400000;
  RadiationScene scene(m);
  RadiationSettings s;
  s.photons_per_face = N;
  s.emit_sources = false;
  auto r = sample_radiation(m, scene, m.initial_temperature, 0.0, 0, s);
  auto G = [&](int a, int b) {
    double f = 0;
    for (auto [el, v] : r.absorbed[static_cast<size_t>(m.elements[static_cast<size_t>(a)].faces[0])])
      if (el == b) f += v;
    return f;
  };
  for (int e = 0; e < 6; ++e) {
    double sum = 0;
    for (int b = 0; b < 6; ++b) sum += G(e, b);
    EXPECT_NEAR(sum, 1.0, 1e-12);
  }
  for (auto [a, b] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{3, 4}, std::pair{2, 5}}) {
    const double g = 0.5 * (G(a, b) + G(b, a));
    EXPECT_NEAR(G(a, b), G(b, a), 4 * std::sqrt(2 * g * (1 - g) / N)) << a << " " << b;
  }
  EXPECT_EQ(r.bounce_limit_hits, 0);
}

TEST(Radiation, TransparentShellLetsEverythingThrough) {
  const std::string glass = card("FTOP", {"3", "0.", "0.", "0.", "1.", "0."}) +
                            card("MEDIUM1", {"12", "1.", "0.", "1.", "1000."}) + ",3,3\n";
  std::string bdf = ref::plate(1, 1, 1.0, 1.0, 0.0, 10, 0, 0) + ref::plate(1, 1, 1.0, 1.0, 1.0, 10, 4, 1, true) +
                    ref::plate(1, 1, 1.0, 1.0, 0.5, 12, 8, 2) + kProps + card("PSHELL", {"12", "1", ".001"});
  Model m = ref::model_from_text(bdf, top(10, glass));
  const int N = 200000;
  const double F = ref::F_parallel_rectangles(1, 1, 1.0);
  EXPECT_NEAR(fraction(m, 0, 1, N), F, tol(F, N));
  EXPECT_EQ(fraction(m, 0, 2, N), 0.0);
}

TEST(Radiation, ExternalSourceAndShadow) {
  // Sun-like source on the z axis far away: flux 1374 W/m^2 on the upper plate, the plate below
  // is in its shadow (dissertation, figs. 3.11-3.14).
  const double D = 1.5e11, flux = 1374.0;
  const double Q = flux * 4 * M_PI * D * D;
  std::string bdf = ref::plate(1, 1, 1.0, 1.0, 0.1, 10, 0, 0) + ref::plate(1, 1, 1.0, 1.0, 0.0, 10, 4, 1, true) + kProps;
  Model m = ref::model_from_text(bdf, top(10, card("RSOURCE1", {"1", "0.5", "0.5", num(D), num(Q), "5788.", "", "200000"})));
  RadiationScene scene(m);
  RadiationSettings s;
  s.emit_faces = false;
  auto r = sample_radiation(m, scene, m.initial_temperature, 0.0, 0, s);
  // Fraction of cone rays hitting the plate: A / (pi R^2) for a far source.
  const double R = scene.radius();
  const double p = 1.0 / (M_PI * R * R);
  const double sigma_rel = std::sqrt((1 - p) / (p * 200000));
  EXPECT_NEAR(r.external[0] / flux, 1.0, 4 * sigma_rel);
  EXPECT_EQ(r.external[1], 0.0);
  EXPECT_NEAR(r.source_power_aimed, flux * M_PI * R * R, 1e-6 * flux);
}

TEST(Radiation, ResultDoesNotDependOnThreads) {
  Model m = parallel(1.0, 1.0, 0.3, "", 2000);
  RadiationScene scene(m);
  RadiationSettings s;
  s.photons_per_face = 2000;
  auto run = [&](int threads) {
#ifdef THORIUM2_HAVE_OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    return sample_radiation(m, scene, m.initial_temperature, 0.0, 5, s);
  };
  auto a = run(1), b = run(4);
  ASSERT_EQ(a.absorbed.size(), b.absorbed.size());
  for (size_t f = 0; f < a.absorbed.size(); ++f) EXPECT_EQ(a.absorbed[f], b.absorbed[f]);
  // Another step gives another sample.
  auto c = sample_radiation(m, scene, m.initial_temperature, 0.0, 6, s);
  EXPECT_NE(a.absorbed[0], c.absorbed[0]);
}
