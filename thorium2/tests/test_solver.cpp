#include "reference.h"
#include "thorium2/solver.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace th;
using ref::card;
using ref::num;

namespace {

// White surfaces: nothing emitted or absorbed (eps = 0, diffuse reflectance 1).
const std::string kWhite = card("FTOP", {"1", "1.", "0.", "0.", "0.", "0."});
const std::string kBlack = card("FTOP", {"1", "0.", "0.", "1.", "0.", "1."});

std::string params(double length, double dt, double T0, int photons = 100) {
  return card("I_PARAM", {"BDF2", "0.", num(length), num(dt), std::to_string(photons)}) + card("T_ALL", {"1", num(T0)});
}

std::string medium(int id, double lambda, const std::string& c) {
  return card("MEDIUM1", {std::to_string(id), "1.", "0.", num(lambda), c}) + ",1,1,1,1,1,1\n";
}

/// Plate of n x 1 unit squares, 1 cm thick, density 1000: mass 10 kg per element.
std::string plates(int n) {
  return ref::plate(n, 1, n, 1.0, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".01"}) + card("MAT1", {"1", "", "", "", "1000."});
}

void run(Solver& s) {
  while (!s.finished()) s.step();
}

}  // namespace

TEST(Solver, HeatSourceIntoOneElementIsExact) {
  Model m = ref::model_from_text(plates(1), kWhite + params(100.0, 10.0, 300.0) + medium(10, 1.0, "500.") +
                                                card("HSOURCE1", {"1", "50."}) + ",1\n");
  Solver s(m);
  run(s);
  EXPECT_NEAR(s.time(), 100.0, 1e-12);
  // dT/dt = Q / (m c) = 50 / (10 * 500) = 0.01 K/s.
  EXPECT_NEAR(s.temperatures()[0], 301.0, 1e-9);
}

TEST(Solver, EnthalpyMakesVariableHeatCapacityExact) {
  // c(T) = 500 + 2 (T - 300), heat input Q t: the enthalpy rises linearly; T solves a quadratic.
  const std::string top = kWhite + params(1000.0, 50.0, 300.0) + medium(10, 1.0, "5") +
                          card("DEPEND", {"5", "2."}) + ",200.\n,300.\n,600.\n,1100.\n" + card("HSOURCE1", {"1", "100."}) + ",1\n";
  Model m = ref::model_from_text(plates(1), top);
  Solver s(m);
  run(s);
  // 10 kg: 10 * [500 dT + (dT)^2] = 100 * 1000  =>  dT^2 + 500 dT - 10000 = 0.
  const double dT = (-500.0 + std::sqrt(500.0 * 500.0 + 4 * 10000.0)) / 2;
  EXPECT_NEAR(s.temperatures()[0], 300.0 + dT, 1e-8);
  // Version 1 balanced c(T) T instead of the enthalpy: 10 [c(T) T - c(300) 300] = 1e5, i.e.
  // 2 T^2 - 100 T - 160000 = 0. It would give a heating of about 9 K instead of 19 K.
  const double T_v1 = (100.0 + std::sqrt(100.0 * 100.0 + 8 * 160000.0)) / 4;
  EXPECT_GT(std::abs((T_v1 - 300.0) - dT), 5.0);
}

TEST(Solver, ConductionRelaxationIsSecondOrder) {
  // Two elements, G = 1 / (0.5/100 + 0.5/100) * 0.01 = 1 W/K, C = 10 kg * 500 = 5000 J/K each:
  // dT = dT0 exp(-2 G t / C).
  auto error_for = [](double dt) {
    Model m = ref::model_from_text(plates(2), kWhite + params(5000.0, dt, 300.0) + medium(10, 100.0, "500.") +
                                                  card("T_INIT", {"1"}) + ",1,400.\n");
    Solver s(m);
    run(s);
    const double expected = 100.0 * std::exp(-2.0 * 1.0 * 5000.0 / 5000.0);
    const double got = s.temperatures()[0] - s.temperatures()[1];
    EXPECT_NEAR(s.temperatures()[0] + s.temperatures()[1], 700.0, 1e-8) << "energy";
    return std::abs(got - expected);
  };
  const double e1 = error_for(250.0), e2 = error_for(125.0), e3 = error_for(62.5);
  EXPECT_LT(e1, 1.0);
  EXPECT_NEAR(std::log2(e1 / e2), 2.0, 0.25) << e1 << " " << e2;
  EXPECT_NEAR(std::log2(e2 / e3), 2.0, 0.15) << e2 << " " << e3;
}

TEST(Solver, StableForHugeSteps) {
  // Copper strip of 10 elements, 1 mm: explicit limit dt < C / (2G) ~ 4.5 s. Take 5000 s.
  const std::string bdf = ref::plate(10, 1, 0.1, 0.01, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".001"}) +
                          card("MAT1", {"1", "", "", "", "8960."});
  Model m = ref::model_from_text(bdf, kWhite + params(50000.0, 5000.0, 300.0) + medium(10, 401.0, "386.") +
                                          card("T_INIT", {"1"}) + ",1,1000.\n");
  Solver s(m);
  double mean0 = 0;
  for (double T : s.temperatures()) mean0 += T;
  while (!s.finished()) {
    s.step();
    for (double T : s.temperatures()) {
      // BDF2 may ring by ~1e-4 K at such steps, but never grows.
      EXPECT_GE(T, 300.0 - 1e-2);
      EXPECT_LE(T, 1000.0 + 1e-2);
    }
  }
  double mean = 0;
  for (double T : s.temperatures()) mean += T;
  EXPECT_NEAR(mean, mean0, 1e-6);
  for (double T : s.temperatures()) EXPECT_NEAR(T, mean0 / 10, 1e-3);
}

TEST(Solver, FixedTemperatureDrivesNeighbour) {
  Model m = ref::model_from_text(plates(2), kWhite + params(200000.0, 2000.0, 300.0) + medium(10, 100.0, "500.") +
                                                card("T_SET", {"1"}) + ",1,350.\n");
  Solver s(m);
  run(s);
  EXPECT_DOUBLE_EQ(s.temperatures()[0], 350.0);
  EXPECT_NEAR(s.temperatures()[1], 350.0, 1e-6);
}

TEST(Solver, RadiativeCoolingOfLonePlate) {
  // Black plate alone: m c dT/dt = -2 sigma A T^4  =>  T = (T0^-3 + 6 sigma A t / (m c))^(-1/3).
  // A lone flat plate sees nothing, so the Monte Carlo part is exact (everything escapes).
  auto error_for = [](double dt) {
    Model m = ref::model_from_text(plates(1), kBlack + params(2000.0, dt, 600.0) + medium(10, 1.0, "500."));
    Solver s(m);
    run(s);
    const double T = std::pow(std::pow(600.0, -3) + 6 * kStefanBoltzmann * 2000.0 / 5000.0, -1.0 / 3);
    return std::abs(s.temperatures()[0] - T);
  };
  const double e1 = error_for(100.0), e2 = error_for(50.0);
  EXPECT_LT(e1, 1.0);  // 20 steps while T falls from 600 K to 192 K
  EXPECT_NEAR(std::log2(e1 / e2), 2.0, 0.3);
  EXPECT_LT(error_for(5.0), 2e-3);
}

TEST(Solver, ClosedBoxConservesEnergyExactly) {
  // Six black plates forming a closed cube with different temperatures: radiation only moves
  // energy around. Sum of m c T is constant up to the Newton tolerance.
  std::string bdf;
  const char* g[8][3] = {{"0.", "0.", "0."}, {"1.", "0.", "0."}, {"1.", "1.", "0."}, {"0.", "1.", "0."},
                         {"0.", "0.", "1."}, {"1.", "0.", "1."}, {"1.", "1.", "1."}, {"0.", "1.", "1."}};
  for (int i = 0; i < 8; ++i) bdf += card("GRID", {std::to_string(i + 1), "", g[i][0], g[i][1], g[i][2]});
  const int q[6][4] = {{1, 2, 3, 4}, {5, 8, 7, 6}, {1, 5, 6, 2}, {2, 6, 7, 3}, {3, 7, 8, 4}, {4, 8, 5, 1}};
  for (int i = 0; i < 6; ++i)
    bdf += card("CQUAD4", {std::to_string(i + 1), "10", std::to_string(q[i][0]), std::to_string(q[i][1]),
                           std::to_string(q[i][2]), std::to_string(q[i][3])});
  bdf += card("PSHELL", {"10", "1", ".01"}) + card("MAT1", {"1", "", "", "", "1000."});
  // Inner side black, outer side white (no losses). Conduction through edges is on too.
  const std::string top = kBlack + card("FTOP", {"2", "1.", "0.", "0.", "0.", "0."}) + params(3000.0, 100.0, 300.0, 2000) +
                          card("MEDIUM1", {"10", "1.", "0.", "10.", "500."}) + ",1,2\n" + card("T_INIT", {"1"}) +
                          ",1,500.,2,250.\n";
  Model m = ref::model_from_text(bdf, top);
  Solver s(m);
  auto energy = [&] {
    double e = 0;
    for (double T : s.temperatures()) e += T;
    return e;
  };
  const double e0 = energy();
  while (!s.finished()) {
    StepReport r = s.step();
    EXPECT_NEAR(energy(), e0, 1e-6 * e0);
    EXPECT_NEAR(r.balance.emitted, r.balance.absorbed, 1e-9 * r.balance.emitted);
  }
  // Heat flows from hot to cold.
  EXPECT_LT(s.temperatures()[0], 500.0);
  EXPECT_GT(s.temperatures()[1], 250.0);
}

TEST(Solver, TimeDependentHeatSource) {
  // Q(t) = 0 until 100 s, then 50 W: energy after 300 s = 50 * 200 minus the ramp 100..110 s.
  const std::string top = kWhite + params(300.0, 1.0, 300.0) + medium(10, 1.0, "500.") +
                          card("DEPEND", {"7", "2."}) + ",0.,0.\n,0.\n,0.,100.\n,0.\n,0.,110.\n,50.\n,0.,1000.\n,50.\n" +
                          card("HSOURCE1", {"1", "7"}) + ",1\n";
  Model m = ref::model_from_text(plates(1), top);
  Solver s(m, {.second_order = false});
  run(s);
  // Backward Euler takes Q at the end of each step: the ramp contributes sum over steps.
  double E = 0;
  for (int k = 1; k <= 300; ++k) {
    const double t = k;
    E += t <= 100 ? 0.0 : (t >= 110 ? 50.0 : 50.0 * (t - 100) / 10);
  }
  EXPECT_NEAR(s.temperatures()[0], 300.0 + E / 5000.0, 1e-9);
}
