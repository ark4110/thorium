// Verification cases of the dissertation (A. Shaenko, 2011), section 3.2. The reference
// solutions are recomputed here independently: ODEs by RK4 with tiny steps, the rod and the
// plate by Fourier series of the continuous heat equation.
#include "reference.h"
#include "thorium2/solver.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace th;
using ref::card;
using ref::num;

namespace {

constexpr double kS = kStefanBoltzmann;

/// Two black 1 x 1 m Mylar plates (1 mm, 1450 kg/m^3), 0.1 m apart, front faces facing.
std::string two_plates_bdf() {
  return ref::plate(1, 1, 1.0, 1.0, 0.0, 10, 0, 0) + ref::plate(1, 1, 1.0, 1.0, 0.1, 10, 4, 1, true) +
         card("PSHELL", {"10", "1", ".001"}) + card("MAT1", {"1", "", "", "", "1450."});
}

std::string plates_top(double length, double dt, int photons, const std::string& c = "1000.", const std::string& extra = "") {
  return card("FTOP", {"1", "0.", "0.", "1.", "0.", "1."}) + card("MEDIUM1", {"10", "1.", "0.", "0.24", c}) + ",1,1\n" +
         card("I_PARAM", {"BDF2", "0.", num(length), num(dt), std::to_string(photons)}) + extra;
}

void run(Solver& s) {
  while (!s.finished()) s.step();
}

const double F01 = ref::F_parallel_rectangles(1, 1, 0.1);  // 0.8269

}  // namespace

TEST(Dissertation, Fig3_7_CoolingOfTwoPlates) {
  // Both plates at 300 K; each emits from two sides and gets F of the other's inner side:
  // C dT/dt = -sigma (2 - F) T^4, C = 1.45 kg * 1000 J/(kg K).
  Model m = ref::model_from_text(two_plates_bdf(), plates_top(100.0, 10.0, 20000, "1000.", card("T_ALL", {"1", "300."})));
  Solver s(m);
  run(s);
  const double C = 1450.0;
  const double T = std::pow(std::pow(300.0, -3) + 3 * kS * (2 - F01) * 100.0 / C, -1.0 / 3);
  EXPECT_NEAR(s.temperatures()[0], T, 0.1);
  EXPECT_NEAR(s.temperatures()[1], T, 0.1);
  EXPECT_NEAR(T, 270.0, 3.0);  // as in fig. 3.7
}

TEST(Dissertation, Fig3_9_PlateHeatedByPlateAtFixedTemperature) {
  // Plate 1 held at 300 K, plate 2 starts at 200 K: C dT2/dt = sigma (F 300^4 - 2 T2^4).
  Model m = ref::model_from_text(two_plates_bdf(),
                                 plates_top(3000.0, 10.0, 5000, "1000.",
                                            card("T_ALL", {"1", "200."}) + card("T_SET", {"1"}) + ",1,300.\n"));
  Solver s(m);
  std::vector<double> y = {200.0};
  double t = 0;
  while (!s.finished()) {
    s.step();
    y = ref::rk4([](double, const std::vector<double>& v) {
      return std::vector<double>{kS * (F01 * std::pow(300.0, 4) - 2 * std::pow(v[0], 4)) / 1450.0};
    }, y, t, s.time(), 100);
    t = s.time();
    if (std::fmod(t, 500.0) < 1e-9) {
      EXPECT_NEAR(s.temperatures()[1], y[0], 0.3) << "t = " << t;
    }
  }
  EXPECT_NEAR(y[0], 300.0 * std::pow(F01 / 2, 0.25), 0.05);  // steady state, ~240 K as in fig. 3.9
}

TEST(Dissertation, Fig3_11_SunOnOnePlate) {
  // Flux 1374 W/m^2 along -z on the upper plate; the lower one is in its shadow.
  const double D = 1.5e11, flux = 1374.0;
  const std::string src = card("RSOURCE1", {"1", "0.5", "0.5", num(D), num(flux * 4 * M_PI * D * D), "5788.", "", "40000"});
  Model m = ref::model_from_text(two_plates_bdf(), plates_top(3000.0, 10.0, 5000, "1000.", card("T_ALL", {"1", "300."}) + src));
  Solver s(m);
  run(s);
  auto f = [&](double, const std::vector<double>& v) {
    const double T1 = v[0], T2 = v[1];  // T1: lower (shadowed), T2: upper (lit)
    return std::vector<double>{kS * (F01 * std::pow(T2, 4) - 2 * std::pow(T1, 4)) / 1450.0,
                               (flux + kS * (F01 * std::pow(T1, 4) - 2 * std::pow(T2, 4))) / 1450.0};
  };
  auto y = ref::rk4(f, {300.0, 300.0}, 0, 3000.0, 30000);
  EXPECT_NEAR(s.temperatures()[1], y[1], 0.5);
  EXPECT_NEAR(s.temperatures()[0], y[0], 0.5);
  EXPECT_GT(y[1], y[0] + 20);
}

TEST(Dissertation, Fig3_16_HeatCapacityDependsOnTemperature) {
  // As fig. 3.9 with c(T) = 1000 (1 + 0.002 (T - 250)); the balance uses the enthalpy.
  const std::string c = card("DEPEND", {"5", "2."}) + ",100.\n,700.\n,400.\n,1300.\n";
  Model m = ref::model_from_text(
      two_plates_bdf(), plates_top(3000.0, 10.0, 5000, "5", card("T_ALL", {"1", "200."}) + card("T_SET", {"1"}) + ",1,300.\n" + c));
  Solver s(m);
  run(s);
  auto y = ref::rk4([](double, const std::vector<double>& v) {
    const double cT = 1000.0 * (1 + 0.002 * (v[0] - 250.0));
    return std::vector<double>{kS * (F01 * std::pow(300.0, 4) - 2 * std::pow(v[0], 4)) / (1.45 * cT)};
  }, {200.0}, 0, 1000.0, 10000);
  // Compare at the end of the run and in the middle of the transient.
  Model m2 = ref::model_from_text(
      two_plates_bdf(), plates_top(1000.0, 10.0, 5000, "5", card("T_ALL", {"1", "200."}) + card("T_SET", {"1"}) + ",1,300.\n" + c));
  Solver s2(m2);
  run(s2);
  EXPECT_NEAR(s2.temperatures()[1], y[0], 0.3);
  EXPECT_NEAR(s.temperatures()[1], 300.0 * std::pow(F01 / 2, 0.25), 0.3);
}

TEST(Dissertation, Fig3_28_CopperRodOf100Hexahedra) {
  // 1 m rod, cross-section 1 cm^2, copper; +1 W in the first element, -1 W in the last, no radiation.
  const int n = 100;
  const double L = 1.0, side = 0.01, A = side * side, h = L / n;
  const double rho = 8960, c = 386, lambda = 401, q = 1.0, alpha = lambda / (rho * c);
  const std::string bdf = ref::hex_rod(n, L, side, 20) + card("PSOLID", {"20", "1"}) + card("MAT1", {"1", "", "", "", num(rho)});
  for (double t_end : {2000.0, 20000.0}) {
    const std::string top = card("FTOP", {"1", "1.", "0.", "0.", "0.", "0."}) +
                            card("MEDIUM1", {"20", "1.", "0.", num(lambda), num(c)}) + ",1,1,1,1,1,1\n" +
                            card("I_PARAM", {"BDF2", "0.", num(t_end), "10.", "10"}) + card("T_ALL", {"1", "300."}) +
                            card("HSOURCE1", {"1", "1."}) + ",1\n" + card("HSOURCE1", {"2", "-1."}) + ",100\n";
    Model m = ref::model_from_text(bdf, top);
    Solver s(m);
    run(s);
    // Fourier series of u_t = alpha u_xx + s(x)/(rho c) with insulated ends, averaged over elements.
    double max_err = 0;
    for (int i = 0; i < n; ++i) {
      const double xa = i * h, xb = xa + h;
      double u = 300.0;
      for (int k = 1; k <= 4000; ++k) {
        const double kk = k * M_PI / L;
        const double sn = (2 / L) * (q / (A * h)) * (std::sin(kk * h) / kk) * (1 - ((k % 2) ? -1.0 : 1.0));
        if (sn == 0) continue;
        const double bn = sn / (rho * c * alpha * kk * kk) * (1 - std::exp(-alpha * kk * kk * t_end));
        u += bn * (std::sin(kk * xb) - std::sin(kk * xa)) / (kk * h);
      }
      max_err = std::max(max_err, std::abs(s.temperatures()[static_cast<size_t>(i)] - u));
    }
    // Spread of temperatures is ~25 K at the steady state.
    EXPECT_LT(max_err, 0.05) << "t = " << t_end;
  }
}

namespace {
/// Fig. 3.30: 1 x 1 x 0.01 m copper plate of n x n CQUAD4, +25 W spread over the corner square
/// 0.1 x 0.1 m, -25 W over the opposite one. Returns the largest difference from the series
/// solution of the continuous problem (element averages) and the largest heating.
std::pair<double, double> copper_plate_error(int n) {
  const double h = 1.0 / n, th = 0.01, rho = 8960, c = 386, lambda = 401, q = 25.0, alpha = lambda / (rho * c);
  const double t_end = 3000.0, a = 0.1;  // source square side
  const int ns = static_cast<int>(std::lround(a / h));
  const std::string bdf = ref::plate(n, n, 1.0, 1.0, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", num(th)}) +
                          card("MAT1", {"1", "", "", "", num(rho)});
  std::string hot = card("HSOURCE1", {"1", num(q)}), cold = card("HSOURCE1", {"2", num(-q)});
  std::vector<std::string> hl, cl;
  for (int j = 0; j < ns; ++j)
    for (int i = 0; i < ns; ++i) {
      hl.push_back(std::to_string(1 + j * n + i));
      cl.push_back(std::to_string(1 + (n - 1 - j) * n + (n - 1 - i)));
    }
  auto list = [](const std::vector<std::string>& v) {
    std::string out;
    for (size_t k = 0; k < v.size(); ++k) out += (k % 8 == 0 ? std::string(k ? "\n," : ",") : std::string(",")) + v[k];
    return out + "\n";
  };
  const std::string top = card("FTOP", {"1", "1.", "0.", "0.", "0.", "0."}) +
                          card("MEDIUM1", {"10", "1.", "0.", num(lambda), num(c)}) + ",1,1\n" +
                          card("I_PARAM", {"BDF2", "0.", num(t_end), "10.", "10"}) + card("T_ALL", {"1", "300."}) + hot +
                          list(hl) + cold + list(cl);
  Model m = ref::model_from_text(bdf, top);
  Solver s(m);
  run(s);
  const int K = 600;
  auto I = [&](int k) { return k == 0 ? a : std::sin(k * M_PI * a) / (k * M_PI); };
  auto avg = [&](int k, int cell) {
    if (k == 0) return 1.0;
    const double kk = k * M_PI;
    return (std::sin(kk * (cell + 1) * h) - std::sin(kk * cell * h)) / (kk * h);
  };
  std::vector<double> u(static_cast<size_t>(n * n), 300.0);
  for (int p = 0; p <= K; ++p)
    for (int r = 0; r <= K; ++r) {
      if ((p + r) % 2 == 0) continue;  // 1 - (-1)^(p+r) = 0
      const double ep = p == 0 ? 1 : 2, er = r == 0 ? 1 : 2;
      const double smn = ep * er * (q / (a * a * th)) * I(p) * I(r) * 2;
      const double k2 = M_PI * M_PI * (p * p + r * r);
      const double bmn = smn / (rho * c * alpha * k2) * (1 - std::exp(-alpha * k2 * t_end));
      for (int j = 0; j < n; ++j) {
        const double ay = avg(r, j) * bmn;
        for (int i = 0; i < n; ++i) u[static_cast<size_t>(j * n + i)] += avg(p, i) * ay;
      }
    }
  double max_err = 0, spread = 0;
  for (size_t k = 0; k < u.size(); ++k) {
    max_err = std::max(max_err, std::abs(s.temperatures()[k] - u[k]));
    spread = std::max(spread, std::abs(u[k] - 300.0));
  }
  EXPECT_NEAR(s.temperatures().front() - 300.0, 300.0 - s.temperatures().back(), 1e-6) << "antisymmetry";
  return {max_err, spread};
}
}  // namespace

TEST(Dissertation, Fig3_30_CopperPlateConvergesToContinuousSolution) {
  // The finite volume scheme is second order: the error must fall about 4 times per refinement.
  const auto [e10, s10] = copper_plate_error(10);
  const auto [e20, s20] = copper_plate_error(20);
  const auto [e40, s40] = copper_plate_error(40);
  EXPECT_GT(s10, 2.0);
  // The 10 x 10 mesh of the dissertation is coarse next to the sources: ~10% of the heating there.
  EXPECT_LT(e10, 0.12 * s10) << e10 << " K of " << s10 << " K";
  EXPECT_LT(e40, 0.01 * s40) << e40 << " K of " << s40 << " K";
  EXPECT_GT(e10 / e20, 3.0) << e10 << " " << e20;
  EXPECT_GT(e20 / e40, 3.0) << e20 << " " << e40;
  std::printf("copper plate, max error: 10x10 %.4f K, 20x20 %.4f K, 40x40 %.4f K (heating up to %.3f K)\n", e10, e20, e40, s40);
}
