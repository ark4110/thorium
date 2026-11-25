#include "reference.h"
#include "thorium2/conduction.h"

#include <gtest/gtest.h>

#include <numeric>

using namespace th;
using ref::card;
using ref::num;

namespace {
const std::string kMat = card("MAT1", {"1", "", "", "", "1000."});

std::string medium(int id, double lambda, double c = 1000.0) {
  return card("MEDIUM1", {std::to_string(id), "1.", "0.", num(lambda), num(c)}) + ",1,1,1,1,1,1\n";
}
std::string top(const std::string& media) { return ref::top_header(1.0, 1.0, 10, 300.0) + media; }

/// Heat flows q for temperatures T.
std::vector<double> flows(const Model& m, const std::vector<double>& T, std::vector<Triplet>* J = nullptr) {
  std::vector<double> q(m.elements.size(), 0.0);
  add_conduction(m, T, element_conductivity(m, T, 0.0), q, J);
  return q;
}
}  // namespace

TEST(Conduction, HexRodIsSeriesOfResistances) {
  // Two 1x1x1 m cubes, lambda = 10: G = A / (0.5/10 + 0.5/10) = 10 W/K.
  Model m = ref::model_from_text(ref::hex_rod(2, 2.0, 1.0, 20) + card("PSOLID", {"20", "1"}) + kMat, top(medium(20, 10.0)));
  auto q = flows(m, {310.0, 300.0});
  EXPECT_NEAR(q[1], 100.0, 1e-9);
  EXPECT_NEAR(q[0], -100.0, 1e-9);
}

TEST(Conduction, DifferentMaterialsUseSeriesNotArithmeticMean) {
  // Copper (401) against Mylar (0.24), equal halves: G = A / (0.5/401 + 0.5/0.24).
  std::string bdf = ref::hex_rod(2, 2.0, 1.0, 20) + card("PSOLID", {"20", "1"}) + card("PSOLID", {"21", "1"}) + kMat;
  // Move the second element to property 21.
  bdf.replace(bdf.find("CHEXA,2,20"), 10, "CHEXA,2,21");
  Model m = ref::model_from_text(bdf, top(medium(20, 401.0) + medium(21, 0.24)));
  const double G = 1.0 / (0.5 / 401.0 + 0.5 / 0.24);
  auto q = flows(m, {301.0, 300.0});
  EXPECT_NEAR(q[1], G, 1e-12);
  // The arithmetic mean of version 1 would give ~200 W/K.
  EXPECT_LT(q[1], 0.5);
}

TEST(Conduction, CoplanarShellsWithDifferentThickness) {
  // Two 1x1 plates sharing the edge x = 1; t = 1 mm and 3 mm, lambda = 100 and 50.
  // Each half: g = lambda * t * L / d with d = 0.5 m.
  std::string bdf = ref::plate(2, 1, 2.0, 1.0, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".001"}) +
                    card("PSHELL", {"11", "1", ".003"}) + kMat;
  bdf.replace(bdf.find("CQUAD4,2,10"), 11, "CQUAD4,2,11");
  Model m = ref::model_from_text(bdf, top(medium(10, 100.0) + medium(11, 50.0)));
  const double g1 = 100.0 * 0.001 * 1.0 / 0.5, g2 = 50.0 * 0.003 * 1.0 / 0.5;
  auto q = flows(m, {310.0, 300.0});
  EXPECT_NEAR(q[1], 10.0 * g1 * g2 / (g1 + g2), 1e-12);
}

TEST(Conduction, BentShellsAreConnected) {
  // L-shaped: plate in z=0 and plate in x=1 plane sharing the edge x=1, z=0 (fig. 2.13).
  std::string bdf = card("GRID", {"1", "", "0.", "0.", "0."}) + card("GRID", {"2", "", "1.", "0.", "0."}) +
                    card("GRID", {"3", "", "1.", "1.", "0."}) + card("GRID", {"4", "", "0.", "1.", "0."}) +
                    card("GRID", {"5", "", "1.", "0.", "1."}) + card("GRID", {"6", "", "1.", "1.", "1."}) +
                    card("CQUAD4", {"1", "10", "1", "2", "3", "4"}) + card("CQUAD4", {"2", "10", "2", "5", "6", "3"}) +
                    card("PSHELL", {"10", "1", ".002"}) + kMat;
  Model m = ref::model_from_text(bdf, top(medium(10, 100.0)));
  const double g = 100.0 * 0.002 * 1.0 / 0.5;
  auto q = flows(m, {310.0, 300.0});
  EXPECT_NEAR(q[1], 10.0 * g / 2, 1e-12);
}

TEST(Conduction, TJunctionOfThreeShellsIsAStar) {
  // Three plates meeting on the edge x = 0 (y from 0 to 1): two in z=0 (x<0 and x>0), one vertical.
  std::string bdf = card("GRID", {"1", "", "0.", "0.", "0."}) + card("GRID", {"2", "", "0.", "1.", "0."}) +
                    card("GRID", {"3", "", "1.", "0.", "0."}) + card("GRID", {"4", "", "1.", "1.", "0."}) +
                    card("GRID", {"5", "", "-1.", "0.", "0."}) + card("GRID", {"6", "", "-1.", "1.", "0."}) +
                    card("GRID", {"7", "", "0.", "0.", "1."}) + card("GRID", {"8", "", "0.", "1.", "1."}) +
                    card("CQUAD4", {"1", "10", "1", "3", "4", "2"}) + card("CQUAD4", {"2", "10", "5", "1", "2", "6"}) +
                    card("CQUAD4", {"3", "10", "1", "2", "8", "7"}) + card("PSHELL", {"10", "1", ".001"}) + kMat;
  Model m = ref::model_from_text(bdf, top(medium(10, 100.0)));
  ASSERT_EQ(m.junctions.size(), 1u);
  EXPECT_EQ(m.junctions[0].members.size(), 3u);
  const std::vector<double> T = {330.0, 300.0, 300.0};
  auto q = flows(m, T);
  // Equal g: the junction temperature is the mean, 310 K; q_a = g (310 - T_a).
  const double g = 100.0 * 0.001 * 1.0 / 0.5;
  EXPECT_NEAR(q[0], g * (310.0 - 330.0), 1e-12);
  EXPECT_NEAR(q[1], g * (310.0 - 300.0), 1e-12);
  EXPECT_NEAR(q[2], g * (310.0 - 300.0), 1e-12);
}

TEST(Conduction, ShellLyingOnSolidFace) {
  // Unit cube (lambda 10) with a 1 mm plate (lambda 100) on its top face z = 1.
  std::string bdf = ref::hex_rod(1, 1.0, 1.0, 20) + card("PSOLID", {"20", "1"}) + kMat;
  // Top face of the cube: grids with z = 1 are 3,4,7,8 (y,z)=(1,1),(0,1).
  bdf += card("CQUAD4", {"2", "10", "4", "3", "7", "8"}) + card("PSHELL", {"10", "1", ".001"});
  Model m = ref::model_from_text(bdf, top(medium(20, 10.0) + medium(10, 100.0)));
  ASSERT_EQ(m.junctions.size(), 1u);
  const double G = 1.0 / (0.5 / 10.0 + 0.0005 / 100.0);
  auto q = flows(m, {300.0, 310.0});
  EXPECT_NEAR(q[0], 10.0 * G, 1e-9);
  // The cube's top face and the shell side facing the cube do not radiate.
  int covered = 0;
  for (const Face& f : m.faces) covered += !f.radiating;
  EXPECT_EQ(covered, 2);
  const Element& shell = m.elements[1];
  for (int fi : shell.faces) {
    const Face& f = m.faces[static_cast<size_t>(fi)];
    EXPECT_EQ(f.radiating, f.normal.z > 0) << "only the side facing away from the cube radiates";
  }
}

TEST(Conduction, PlateGridConservesEnergyAndJacobianIsExact) {
  std::string bdf = ref::plate(4, 3, 1.0, 0.75, 0.0, 10, 0, 0) + card("PSHELL", {"10", "1", ".01"}) + kMat;
  Model m = ref::model_from_text(bdf, top(medium(10, 401.0)));
  std::vector<double> T(m.elements.size());
  for (size_t i = 0; i < T.size(); ++i) T[i] = 300.0 + 7.0 * std::sin(static_cast<double>(i));
  std::vector<Triplet> J;
  auto q = flows(m, T, &J);
  EXPECT_NEAR(std::accumulate(q.begin(), q.end(), 0.0), 0.0, 1e-9);
  // Linear in T at fixed conductivity: J * dT reproduces the change of q.
  std::vector<double> dT(T.size()), T2(T);
  for (size_t i = 0; i < T.size(); ++i) T2[i] += dT[i] = 0.01 * static_cast<double>(i % 5);
  auto q2 = flows(m, T2);
  std::vector<double> Jd(T.size(), 0.0);
  for (const Triplet& t : J) Jd[static_cast<size_t>(t.row)] += t.value * dT[static_cast<size_t>(t.col)];
  for (size_t i = 0; i < T.size(); ++i) EXPECT_NEAR(q2[i] - q[i], Jd[i], 1e-9);
  // 4x3 grid: 3*3 vertical + 4*2 horizontal inner edges.
  EXPECT_EQ(m.junctions.size(), 17u);
}

TEST(Conduction, TemperatureDependentConductivity) {
  std::string bdf = ref::hex_rod(2, 2.0, 1.0, 20) + card("PSOLID", {"20", "1"}) + kMat;
  std::string t = ref::top_header(1.0, 1.0, 10, 300.0) + card("MEDIUM1", {"20", "1.", "0.", "3", "1000."}) +
                  ",1,1,1,1,1,1\n" + card("DEPEND", {"3", "2."}) + ",200.\n,10.\n,400.\n,30.\n";
  Model m = ref::model_from_text(bdf, t);
  // lambda(350) = 25, lambda(250) = 15: G = 1 / (0.5/25 + 0.5/15).
  auto q = flows(m, {350.0, 250.0});
  EXPECT_NEAR(q[1], 100.0 / (0.5 / 25.0 + 0.5 / 15.0), 1e-9);
}
