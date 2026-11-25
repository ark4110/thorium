#include "thorium2/top.h"

#include <gtest/gtest.h>

#include <sstream>

using namespace th;

namespace {
TopModel parse(const std::string& text) {
  std::istringstream in(text);
  return parse_top(read_cards(in, "m.top"));
}
}  // namespace

TEST(Top, FtopConstantsAndDefaultEmissivity) {
  TopModel m = parse(
      "FTOP    1       0.95    0.0     0.05    0.0\n"
      "FTOP    2       0.76    0.0     0.24    0.0     0.85\n"
      "FTOP    3       5       0.0     6       0.0     7\n");
  const TopFtop& a = m.ftop.at(1);
  EXPECT_FALSE(a.eps_given);
  EXPECT_DOUBLE_EQ(a.r_dif.constant, 0.95);
  EXPECT_DOUBLE_EQ(a.absorb.constant, 0.05);
  EXPECT_TRUE(m.ftop.at(2).eps_given);
  EXPECT_DOUBLE_EQ(m.ftop.at(2).eps.constant, 0.85);
  const TopFtop& c = m.ftop.at(3);
  EXPECT_TRUE(c.r_dif.is_dependency);
  EXPECT_EQ(c.r_dif.dependency, 5);
  EXPECT_TRUE(c.absorb.is_dependency);
  EXPECT_EQ(c.eps.dependency, 7);
}

TEST(Top, SpectralFtopIsRejected) {
  // CDF / T_eff (fields 8, 9) mean a given emission spectrum: not supported.
  EXPECT_THROW(parse("FTOP    1       0.0     0.0     1.0     0.0     1.0     3\n"), InputError);
}

TEST(Top, Medium) {
  TopModel m = parse(
      "MEDIUM1 1       1.1     1E-5    110.    900.\n"
      "        12      12      1       1       1       1\n");
  const TopMedium& md = m.media.at(1);
  EXPECT_DOUBLE_EQ(md.lambda.constant, 110.0);
  EXPECT_DOUBLE_EQ(md.c.constant, 900.0);
  EXPECT_EQ(md.ftop[0], 12);
  EXPECT_EQ(md.ftop[1], 12);
  EXPECT_EQ(md.ftop[5], 1);
}

TEST(Top, DependFromMoonRover) {
  TopModel m = parse(
      "DEPEND  1       10.\n"
      "        0.      0.\n"
      "        12.5\n"
      "        0.      7200.\n"
      "        40.0\n");
  const TopDepend& d = m.depend.at(1);
  EXPECT_DOUBLE_EQ(d.idw_power, 10.0);
  ASSERT_EQ(d.args.size(), 2u);
  EXPECT_DOUBLE_EQ(*d.args[1][kArgTime], 7200.0);
  EXPECT_DOUBLE_EQ(*d.values[1][0], 40.0);
}

TEST(Top, DependNeedsPairsOfLines) {
  EXPECT_THROW(parse("DEPEND  1       2.\n        0.      0.\n"), InputError);
}

TEST(Top, SourcesTemperaturesAndParameters) {
  TopModel m = parse(
      "RSOURCE1 1      1.05E+110.00E+101.05E+113.85E+265788.0          1000\n"
      "RSOURCE1 2      4                       100.    5788.\n"
      "HSOURCE1 1      1\n"
      "        1       2       3       4       5       6\n"
      "        51      52\n"
      "HSOURCE1 2      -25.\n"
      "        7\n"
      "T_ALL   1       325.\n"
      "T_INIT  1\n"
      "        5       200.    6       210.\n"
      "T_SET   1\n"
      "        10001   325.\n"
      "        31109   9\n"
      "G_MEDIUM       2\n"
      "I_PARAM EULER_V 0.0     10800.  1.0     1000    0.900   1.100\n");
  const TopRsource& s = m.rsources.at(1);
  EXPECT_DOUBLE_EQ(s.x.constant, 1.05e11);
  EXPECT_DOUBLE_EQ(s.power.constant, 3.85e26);
  EXPECT_EQ(s.n_photons, 1000);
  EXPECT_TRUE(m.rsources.at(2).position_is_dependency);
  EXPECT_EQ(m.rsources.at(2).position_dependency, 4);
  EXPECT_EQ(m.hsources.at(1).power.dependency, 1);
  EXPECT_EQ(m.hsources.at(1).elements, (std::vector<int>{1, 2, 3, 4, 5, 6, 51, 52}));
  EXPECT_DOUBLE_EQ(m.hsources.at(2).power.constant, -25.0);
  EXPECT_DOUBLE_EQ(*m.t_all, 325.0);
  ASSERT_EQ(m.t_init.size(), 1u);
  EXPECT_EQ(m.t_init[0].temperatures[1].first, 6);
  EXPECT_DOUBLE_EQ(m.t_init[0].temperatures[1].second, 210.0);
  ASSERT_EQ(m.t_set[0].temperatures.size(), 2u);
  EXPECT_TRUE(m.t_set[0].temperatures[1].second.is_dependency);
  EXPECT_EQ(*m.g_medium, 2);
  ASSERT_TRUE(m.iparam);
  EXPECT_EQ(m.iparam->type, "EULER_V");
  EXPECT_DOUBLE_EQ(m.iparam->t_length, 10800.0);
  EXPECT_DOUBLE_EQ(m.iparam->t_step, 1.0);
  EXPECT_EQ(m.iparam->n_photons, 1000);
}

TEST(Top, Errors) {
  EXPECT_THROW(parse("FTOPP   1       0.95    0.0     0.05    0.0\n"), InputError);           // unknown card
  EXPECT_THROW(parse("FTOP    1       0.95    0.0     0.05    0.0\nFTOP    1       0.95    0.0     0.05    0.0\n"),
               InputError);                                                                     // duplicate
  EXPECT_THROW(parse("I_PARAM EULER   0.0     100.    0.0     1000\n"), InputError);            // zero step
  EXPECT_THROW(parse("I_PARAM RK45    0.0     100.    1.0     1000\n"), InputError);            // unknown type
  EXPECT_THROW(parse("T_INIT  1\n        5       200.    6\n"), InputError);                    // half pair
  EXPECT_THROW(parse("RSOURCE1 1      0.      0.      1.      1.      5788.   7       1000\n"), InputError);  // spectrum
}

TEST(Top, ReadsV1ExampleFiles) {
  for (const char* name : {"Distorsion_test", "Moon_rover", "Space_telescope_1", "Space_telescope_2"}) {
    const std::string path = std::string(THORIUM2_TEST_DATA) + "/v1_examples/" + name + "/" + name + ".top";
    TopModel m = read_top_file(path);
    EXPECT_FALSE(m.ftop.empty()) << name;
    EXPECT_FALSE(m.media.empty()) << name;
    EXPECT_TRUE(m.iparam.has_value()) << name;
    EXPECT_TRUE(m.t_all.has_value()) << name;
  }
  TopModel rover = read_top_file(std::string(THORIUM2_TEST_DATA) + "/v1_examples/Moon_rover/Moon_rover.top");
  EXPECT_EQ(rover.depend.at(1).args.size(), 22u);
  EXPECT_EQ(rover.t_set.at(0).temperatures.size(), 9u);
}
