// The four example models shipped with T.H.O.R.I.U.M. 1 run in Thorium 2.
#include "thorium2/solver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using namespace th;

TEST(V1Examples, EveryModelRunsAndItsPowerBalanceCloses) {
  for (const char* name : {"Distorsion_test", "Moon_rover", "Space_telescope_1", "Space_telescope_2"}) {
    Model m = load_model(std::string(THORIUM2_TEST_DATA) + "/v1_examples/" + name + "/" + name + ".bdf");
    // Two steps of backward Euler with few photons: with BE the balance closes exactly.
    Solver s(m, {.photons_per_face = 100, .second_order = false});
    for (int k = 0; k < 2; ++k) {
      StepReport r = s.step();
      for (double T : s.temperatures()) ASSERT_TRUE(std::isfinite(T) && T > 0) << name;
      const PowerBalance& b = r.balance;
      const double in = b.external + b.heat_sources + b.absorbed - b.emitted;
      const double scale = std::max({1.0, b.emitted, std::abs(b.external), std::abs(b.heat_sources)});
      EXPECT_NEAR(b.stored_rate + b.into_fixed, in, 1e-6 * scale) << name;
      EXPECT_LE(r.newton_iterations, 10) << name;
      EXPECT_EQ(r.cuts, 0) << name;
    }
  }
}
