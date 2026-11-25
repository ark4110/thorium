#include "thorium2/bdf.h"

#include <gtest/gtest.h>

#include <sstream>

using namespace th;

namespace {
BdfModel parse(const std::string& text) {
  std::istringstream in(text);
  return parse_bdf(read_cards(in, "m.bdf"));
}

const char* kSmall =
    "GRID    1               0.      0.      0.\n"
    "GRID    2               1.      0.      0.\n"
    "GRID    3               1.      1.      0.\n"
    "GRID    4               0.      1.      0.\n"
    "GRID    5               0.      0.      1.\n"
    "MAT1    7       7.+10           .3      2700.\n"
    "PSHELL  10      7       .005                                    0.5\n"
    "PSOLID  20      7\n"
    "CQUAD4  100     10      1       2       3       4\n"
    "CTRIA3  101     10      1       2       3\n"
    "CTETRA  102     20      1       2       4       5\n";
}  // namespace

TEST(Bdf, ReadsGridsElementsAndProperties) {
  BdfModel m = parse(kSmall);
  ASSERT_EQ(m.grids.size(), 5u);
  EXPECT_DOUBLE_EQ(m.grids.at(3).x.y, 1.0);
  ASSERT_EQ(m.elements.size(), 3u);
  EXPECT_EQ(m.elements[0].type, ElementType::CQUAD4);
  EXPECT_EQ(m.elements[0].pid, 10);
  EXPECT_EQ(m.elements[0].grids, (std::vector<int>{1, 2, 3, 4}));
  EXPECT_EQ(m.elements[2].type, ElementType::CTETRA);
  EXPECT_EQ(m.pshell.at(10).mid, 7);
  EXPECT_DOUBLE_EQ(m.pshell.at(10).thickness, 0.005);
  EXPECT_DOUBLE_EQ(m.pshell.at(10).nsm, 0.5);
  EXPECT_EQ(m.psolid.at(20).mid, 7);
  EXPECT_DOUBLE_EQ(m.materials.at(7).rho, 2700.0);
}

TEST(Bdf, HexaPentaWithContinuationAndMidsideNodes) {
  BdfModel m = parse(
      "CHEXA   1       2       1       2       3       4       5       6\n"
      "+       7       8\n"
      "CPENTA  2       2       1       2       3       4       5       6\n"
      "CTETRA  3       2       1       2       3       4       5       6       7\n"
      "+       8       9       10\n");
  ASSERT_EQ(m.elements.size(), 3u);
  EXPECT_EQ(m.elements[0].grids, (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(m.elements[1].grids.size(), 6u);
  // CTETRA10: midside nodes dropped, with a warning.
  EXPECT_EQ(m.elements[2].grids, (std::vector<int>{1, 2, 3, 4}));
  EXPECT_FALSE(m.warnings.empty());
}

TEST(Bdf, HigherOrderShellsUseCorners) {
  BdfModel m = parse(
      "CTRIA6  1       2       1       2       3       4       5       6\n"
      "CQUAD8  2       2       1       2       3       4       5       6\n"
      "+       7       8\n");
  ASSERT_EQ(m.elements.size(), 2u);
  EXPECT_EQ(m.elements[0].type, ElementType::CTRIA3);
  EXPECT_EQ(m.elements[0].grids, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(m.elements[1].type, ElementType::CQUAD4);
  EXPECT_EQ(m.elements[1].grids, (std::vector<int>{1, 2, 3, 4}));
}

TEST(Bdf, LargeFieldGrid) {
  BdfModel m = parse(
      "GRID*                  7               0  1.25000000E+00  2.50000000E+00\n"
      "*        -3.7500000E+00               0\n");
  EXPECT_DOUBLE_EQ(m.grids.at(7).x.z, -3.75);
}

TEST(Bdf, IgnoredCardsAreCounted) {
  BdfModel m = parse(std::string(kSmall) + "FORCE   1       1       0       1.      0.      0.      1.\nSPC1    1       123456  1\n");
  EXPECT_EQ(m.ignored.at("FORCE"), 1);
  EXPECT_EQ(m.ignored.at("SPC1"), 1);
}

TEST(Bdf, RejectsWhatWouldGiveWrongResults) {
  // Non-basic coordinate system.
  EXPECT_THROW(parse("GRID    1       5       0.      0.      0.\n"), InputError);
  // Coordinate system cards.
  EXPECT_THROW(parse("CORD2R  1               0.      0.      0.      0.      0.      1.\n"), InputError);
  // Duplicate grid / element ids.
  EXPECT_THROW(parse("GRID    1               0.      0.      0.\nGRID    1               1.      0.      0.\n"), InputError);
  EXPECT_THROW(parse("CTRIA3  1       2       1       2       3\nCTRIA3  1       2       1       2       3\n"), InputError);
  // Missing corner grid.
  EXPECT_THROW(parse("CTRIA3  1       2       1       2\n"), InputError);
  // Repeated grid in one element.
  EXPECT_THROW(parse("CTRIA3  1       2       1       2       2\n"), InputError);
  // Composite shells are not supported.
  EXPECT_THROW(parse("PCOMP   1\n"), InputError);
  // Elements we do not model thermally would silently lose heat paths.
  EXPECT_THROW(parse("CBAR    1       2       1       2\n"), InputError);
  EXPECT_THROW(parse("CROD    1       2       1       2\n"), InputError);
  // Non-positive thickness or density.
  EXPECT_THROW(parse("PSHELL  1       1       0.\n"), InputError);
  EXPECT_THROW(parse("MAT1    1       7.+10           .3      -1.\n"), InputError);
}

TEST(Bdf, ReadsV1ExampleModels) {
  for (const char* name : {"Distorsion_test", "Moon_rover", "Space_telescope_1", "Space_telescope_2"}) {
    const std::string path = std::string(THORIUM2_TEST_DATA) + "/v1_examples/" + name + "/" + name + ".bdf";
    BdfModel m = read_bdf_file(path);
    EXPECT_GT(m.grids.size(), 100u) << name;
    EXPECT_GT(m.elements.size(), 100u) << name;
    EXPECT_FALSE(m.pshell.empty()) << name;
    EXPECT_FALSE(m.materials.empty()) << name;
  }
  BdfModel rover = read_bdf_file(std::string(THORIUM2_TEST_DATA) + "/v1_examples/Moon_rover/Moon_rover.bdf");
  EXPECT_EQ(rover.elements.size(), 123u + 52u);
  EXPECT_DOUBLE_EQ(rover.materials.at(1).rho, 2700.0);
  EXPECT_DOUBLE_EQ(rover.pshell.at(30002).thickness, 0.1);
}
