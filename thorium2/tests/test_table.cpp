#include "thorium2/table.h"

#include <gtest/gtest.h>

#include <sstream>

using namespace th;

namespace {
Table table(const std::string& text) {
  std::istringstream in(text);
  TopModel m = parse_top(read_cards(in, "t.top"));
  return Table(m.depend.begin()->second);
}
}  // namespace

TEST(Table, LinearInTemperatureIsExact) {
  // c(T) = 386 * (1 + 0.005 (T - 300)), the copper of fig. 3.26 of the dissertation.
  Table t = table(
      "DEPEND  1       2.\n"
      "        290.\n        366.7\n"
      "        310.\n        405.3\n");
  EXPECT_TRUE(t.depends_on_temperature());
  EXPECT_FALSE(t.depends_on_time());
  EXPECT_NEAR(t.eval(300.0, 0.0), 386.0, 1e-12);
  EXPECT_NEAR(t.eval(295.0, 123.0), 376.35, 1e-12);
  // Constant beyond the ends.
  EXPECT_NEAR(t.eval(200.0, 0.0), 366.7, 1e-12);
  EXPECT_NEAR(t.eval(400.0, 0.0), 405.3, 1e-12);
}

TEST(Table, IntegralIsExactForPiecewiseLinear) {
  Table t = table(
      "DEPEND  1       2.\n"
      "        100.\n        1.\n"
      "        200.\n        3.\n"
      "        300.\n        3.\n");
  // From 50 to 350: 50*1 (below) + (1+3)/2*100 + 3*100 + 50*3 (above).
  EXPECT_NEAR(t.integrate_temperature(50.0, 350.0), 50.0 + 200.0 + 300.0 + 150.0, 1e-9);
  // Reversed limits: -(c(150) + c(125)) / 2 * 25 = -(2 + 1.5) / 2 * 25.
  EXPECT_NEAR(t.integrate_temperature(150.0, 125.0), -43.75, 1e-9);
  EXPECT_NEAR(t.integrate_temperature(120.0, 120.0), 0.0, 1e-12);
}

TEST(Table, TimeDependencyWrittenWithZeroTemperature) {
  // Moon_rover.top: temperature column is "0." in every point, so only time is an argument.
  Table t = table(
      "DEPEND  1       10.\n"
      "        0.      0.\n        12.5\n"
      "        0.      7199.\n        12.5\n"
      "        0.      7200.\n        40.0\n");
  EXPECT_FALSE(t.depends_on_temperature());
  EXPECT_TRUE(t.depends_on_time());
  EXPECT_DOUBLE_EQ(t.eval(500.0, 3000.0), 12.5);
  EXPECT_DOUBLE_EQ(t.eval(0.0, 7199.5), 26.25);
  EXPECT_DOUBLE_EQ(t.eval(0.0, 9000.0), 40.0);
}

TEST(Table, BilinearInTemperatureAndTime) {
  Table t = table(
      "DEPEND  1       2.\n"
      "        100.    0.\n        1.\n"
      "        200.    0.\n        2.\n"
      "        100.    10.\n        3.\n"
      "        200.    10.\n        6.\n");
  EXPECT_TRUE(t.depends_on_temperature());
  EXPECT_TRUE(t.depends_on_time());
  EXPECT_DOUBLE_EQ(t.eval(150.0, 0.0), 1.5);
  EXPECT_DOUBLE_EQ(t.eval(150.0, 10.0), 4.5);
  EXPECT_DOUBLE_EQ(t.eval(150.0, 5.0), 3.0);
}

TEST(Table, VectorValues) {
  Table t = table(
      "DEPEND  4       2.\n"
      "        0.      0.\n        1.      2.      3.\n"
      "        0.      10.\n        11.     12.     13.\n");
  EXPECT_EQ(t.value_components(), 3);
  EXPECT_DOUBLE_EQ(t.eval(0.0, 5.0, 2), 8.0);
}

TEST(Table, Rejected) {
  // Two arguments, not a rectangular grid.
  EXPECT_THROW(table("DEPEND  1       2.\n"
                     "        100.    0.\n        1.\n"
                     "        200.    5.\n        2.\n"
                     "        150.    10.\n        3.\n"),
               InputError);
  // Frequency argument (spectral properties) is not supported.
  EXPECT_THROW(table("DEPEND  1       2.\n"
                     "        0.      0.              1.E14\n        1.\n"
                     "        0.      0.              2.E14\n        2.\n"),
               InputError);
  // Same argument twice.
  EXPECT_THROW(table("DEPEND  1       2.\n"
                     "        100.\n        1.\n"
                     "        100.\n        2.\n"),
               InputError);
  // An argument missing in one point.
  EXPECT_THROW(table("DEPEND  1       2.\n"
                     "        100.    1.\n        1.\n"
                     "                2.\n        2.\n"),
               InputError);
}
