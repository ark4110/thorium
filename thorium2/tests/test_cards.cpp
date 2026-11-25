#include "thorium2/cards.h"

#include <gtest/gtest.h>

#include <sstream>

using namespace th;

namespace {
std::vector<Card> read(const std::string& text) {
  std::istringstream in(text);
  return read_cards(in, "test.bdf");
}
}  // namespace

TEST(Numbers, NastranRealForms) {
  EXPECT_DOUBLE_EQ(*parse_real("1.5"), 1.5);
  EXPECT_DOUBLE_EQ(*parse_real(".5"), 0.5);
  EXPECT_DOUBLE_EQ(*parse_real("5."), 5.0);
  EXPECT_DOUBLE_EQ(*parse_real("7.+10"), 7e10);
  EXPECT_DOUBLE_EQ(*parse_real("1.5-3"), 1.5e-3);
  EXPECT_DOUBLE_EQ(*parse_real("-2.5+2"), -250.0);
  EXPECT_DOUBLE_EQ(*parse_real("1.05E+11"), 1.05e11);
  EXPECT_DOUBLE_EQ(*parse_real("3E9"), 3e9);
  EXPECT_DOUBLE_EQ(*parse_real("2.D3"), 2000.0);
  EXPECT_DOUBLE_EQ(*parse_real("12"), 12.0);
  EXPECT_FALSE(parse_real("").has_value());
  EXPECT_FALSE(parse_real("abc").has_value());
  EXPECT_FALSE(parse_real("1.2.3").has_value());
  EXPECT_FALSE(parse_real("1e").has_value());
}

TEST(Numbers, IntegerLiteralMeansDependency) {
  EXPECT_TRUE(is_integer_literal("12"));
  EXPECT_TRUE(is_integer_literal("-3"));
  EXPECT_FALSE(is_integer_literal("12."));
  EXPECT_FALSE(is_integer_literal("1E3"));
  EXPECT_FALSE(is_integer_literal(""));
  EXPECT_EQ(*parse_integer("31302"), 31302);
  EXPECT_FALSE(parse_integer("1.").has_value());
}

TEST(Cards, SmallFieldWithComments) {
  auto cards = read(
      "$ comment\n"
      "\n"
      "GRID     31001          0.175   -0.075   0.350\n"
      "CQUAD4   31001   30001   31002   31001   31201   31202\n");
  ASSERT_EQ(cards.size(), 2u);
  EXPECT_EQ(cards[0].name, "GRID");
  EXPECT_EQ(cards[0].integer(2), 31001);
  EXPECT_TRUE(cards[0].blank(3));
  EXPECT_DOUBLE_EQ(cards[0].real(4), 0.175);
  EXPECT_DOUBLE_EQ(cards[0].real(5), -0.075);
  EXPECT_DOUBLE_EQ(cards[0].real(6), 0.35);
  EXPECT_EQ(cards[0].line, 3);
  EXPECT_EQ(cards[1].integer(7), 31202);
}

TEST(Cards, FieldsAreFixedEightColumnsEvenWithoutSpaces) {
  // From Moon_rover.top: numbers touch each other.
  auto cards = read("RSOURCE1 1      1.05E+110.00E+101.05E+113.85E+265788.0          1000\n");
  ASSERT_EQ(cards.size(), 1u);
  const Card& c = cards[0];
  EXPECT_EQ(c.name, "RSOURCE1");
  EXPECT_EQ(c.integer(2), 1);
  EXPECT_DOUBLE_EQ(c.real(3), 1.05e11);
  EXPECT_DOUBLE_EQ(c.real(4), 0.0);
  EXPECT_DOUBLE_EQ(c.real(5), 1.05e11);
  EXPECT_DOUBLE_EQ(c.real(6), 3.85e26);
  EXPECT_DOUBLE_EQ(c.real(7), 5788.0);
  EXPECT_TRUE(c.blank(8));
  EXPECT_EQ(c.integer(9), 1000);
}

TEST(Cards, ContinuationLines) {
  auto cards = read(
      "MEDIUM1 1       1.1     1E-5    110.    900.\n"
      "        12      12      1       1       1       1\n"
      "HSOURCE1 2      2\n"
      "        31302   32302   33302    34302\n"
      "+CONT   7\n");
  ASSERT_EQ(cards.size(), 2u);
  EXPECT_EQ(cards[0].line_count(), 2);
  EXPECT_EQ(cards[0].field(1, 2), "12");
  EXPECT_EQ(cards[0].field(1, 7), "1");
  EXPECT_EQ(cards[0].field(10), "12");
  EXPECT_EQ(cards[1].line_count(), 3);
  EXPECT_EQ(cards[1].field(1, 5), "34302");
  EXPECT_EQ(cards[1].field(2, 2), "7");
}

TEST(Cards, LargeField) {
  auto cards = read(
      "GRID*                  7               0  1.25000000E+00  2.50000000E+00\n"
      "*        -3.7500000E+00               0\n");
  ASSERT_EQ(cards.size(), 1u);
  EXPECT_EQ(cards[0].name, "GRID");
  EXPECT_EQ(cards[0].integer(2), 7);
  EXPECT_EQ(cards[0].integer(3), 0);
  EXPECT_DOUBLE_EQ(cards[0].real(4), 1.25);
  EXPECT_DOUBLE_EQ(cards[0].real(5), 2.5);
  EXPECT_DOUBLE_EQ(cards[0].real(6), -3.75);
  EXPECT_EQ(cards[0].integer(7), 0);
}

TEST(Cards, FreeField) {
  auto cards = read("GRID,5,,1.,2.,3.\nCTRIA3,9,1,1,2,3\n");
  ASSERT_EQ(cards.size(), 2u);
  EXPECT_EQ(cards[0].integer(2), 5);
  EXPECT_TRUE(cards[0].blank(3));
  EXPECT_DOUBLE_EQ(cards[0].real(6), 3.0);
  EXPECT_EQ(cards[1].integer(6), 3);
}

TEST(Cards, TabsExpandToEightColumnStops) {
  auto cards = read("GRID\t1\t\t0.\t1.\t2.\n");
  ASSERT_EQ(cards.size(), 1u);
  EXPECT_EQ(cards[0].integer(2), 1);
  EXPECT_TRUE(cards[0].blank(3));
  EXPECT_DOUBLE_EQ(cards[0].real(5), 1.0);
}

TEST(Cards, TabOnlyLineIsBlank) {
  auto cards = read("$Medium Number\tn\tk\n\t\nT_ALL   1       300.\n");
  ASSERT_EQ(cards.size(), 1u);
  EXPECT_EQ(cards[0].name, "T_ALL");
}

TEST(Cards, BeginBulkAndEnddata) {
  auto cards = read(
      "SOL 159\nCEND\nTITLE = x\nBEGIN BULK\nGRID    1               0.      0.      0.\n"
      "ENDDATA\nGRID    2               0.      0.      0.\n");
  ASSERT_EQ(cards.size(), 1u);
  EXPECT_EQ(cards[0].integer(2), 1);
}

TEST(Cards, ErrorsNameFileAndLine) {
  auto cards = read("\nGRID    x               0.      0.      0.\n");
  try {
    (void)cards[0].integer(2);
    FAIL();
  } catch (const InputError& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("test.bdf:2"), std::string::npos) << msg;
    EXPECT_NE(msg.find("GRID"), std::string::npos) << msg;
  }
  EXPECT_THROW((void)cards[0].real(3), InputError);  // blank, no default
  EXPECT_DOUBLE_EQ(cards[0].real(3, 4.0), 4.0);
}

TEST(Cards, ContinuationWithoutCardIsError) {
  EXPECT_THROW(read("        1       2\n"), InputError);
}
