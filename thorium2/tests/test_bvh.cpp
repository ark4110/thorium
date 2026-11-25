#include "thorium2/bvh.h"
#include "thorium2/radiation.h"

#include <gtest/gtest.h>

using namespace th;

TEST(Bvh, SingleTriangle) {
  Bvh b({{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, 0, 0}});
  auto h = b.intersect({0.2, 0.2, 1}, {0, 0, -1}, 1e9, -1);
  EXPECT_EQ(h.tri, 0);
  EXPECT_NEAR(h.t, 1.0, 1e-15);
  EXPECT_EQ(b.intersect({0.8, 0.8, 1}, {0, 0, -1}, 1e9, -1).tri, -1);  // outside
  EXPECT_EQ(b.intersect({0.2, 0.2, 1}, {0, 0, 1}, 1e9, -1).tri, -1);   // behind
  EXPECT_EQ(b.intersect({0.2, 0.2, 1}, {0, 0, -1}, 0.5, -1).tri, -1);  // too far
  EXPECT_EQ(b.intersect({0.2, 0.2, 1}, {0, 0, -1}, 1e9, 0).tri, -1);   // skipped element
}

TEST(Bvh, MatchesBruteForceOnRandomSoup) {
  Random r(42);
  std::vector<Bvh::Tri> tris;
  for (int i = 0; i < 2000; ++i) {
    Vec3 c{r.uniform() * 10, r.uniform() * 10, r.uniform() * 10};
    auto jitter = [&] { return Vec3{r.uniform() - 0.5, r.uniform() - 0.5, r.uniform() - 0.5}; };
    tris.push_back({c + jitter(), c + jitter(), c + jitter(), i, i / 3});
  }
  Bvh b(tris);
  int hits = 0;
  for (int i = 0; i < 3000; ++i) {
    Vec3 o{r.uniform() * 12 - 1, r.uniform() * 12 - 1, r.uniform() * 12 - 1};
    Vec3 d = normalized(Vec3{r.uniform() - 0.5, r.uniform() - 0.5, r.uniform() - 0.5});
    const int skip = i % 7 == 0 ? static_cast<int>(r.uniform() * 600) : -1;
    auto h1 = b.intersect(o, d, 1e9, skip);
    auto h2 = b.intersect_brute_force(o, d, 1e9, skip);
    ASSERT_EQ(h1.tri < 0, h2.tri < 0) << i;
    if (h1.tri >= 0) {
      ++hits;
      EXPECT_NEAR(h1.t, h2.t, 1e-12) << i;
    }
  }
  EXPECT_GT(hits, 500);  // enough rays actually hit something
}

TEST(Random, DeterministicAndUniform) {
  Random a(7), b(7);
  double mean = 0;
  for (int i = 0; i < 100000; ++i) {
    const double x = a.uniform();
    ASSERT_EQ(x, b.uniform());
    ASSERT_GE(x, 0.0);
    ASSERT_LT(x, 1.0);
    mean += x;
  }
  EXPECT_NEAR(mean / 100000, 0.5, 4 * std::sqrt(1.0 / 12 / 100000));
  EXPECT_NE(Random::mix(1, 2, 3), Random::mix(1, 2, 4));
}
