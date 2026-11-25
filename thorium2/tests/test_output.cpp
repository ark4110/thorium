#include "reference.h"
#include "thorium2/output.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

using namespace th;

TEST(Output, NumbersAsVersion1WroteThem) {
  // Math.Round(v, 6) + Double.ToString() of .NET.
  EXPECT_EQ(v1_number(300.0), "300");
  EXPECT_EQ(v1_number(0.0), "0");
  EXPECT_EQ(v1_number(-0.0000001), "0");
  EXPECT_EQ(v1_number(12.5), "12.5");
  EXPECT_EQ(v1_number(-25.0), "-25");
  EXPECT_EQ(v1_number(301.12345678), "301.123457");
  EXPECT_EQ(v1_number(0.0000025), "2E-06");     // half to even: 2.5e-6 -> 2e-6
  EXPECT_EQ(v1_number(0.000123), "0.000123");
  EXPECT_EQ(v1_number(0.0000123), "1.2E-05");
  EXPECT_EQ(v1_number(1e7), "10000000");
  EXPECT_EQ(v1_number(3.85e26), "3.85E+26");
}

TEST(Output, ResultFileGroupsLinesByElement) {
  Model m = ref::model_from_text(ref::plate(2, 1, 2.0, 1.0, 0.0, 10, 0, 0) + ref::card("PSHELL", {"10", "1", ".01"}) +
                                     ref::card("MAT1", {"1", "", "", "", "1000."}),
                                 ref::top_header(1.0, 1.0, 10, 300.0) + ref::card("MEDIUM1", {"10", "1.", "0.", "1.", "1."}) +
                                     ",1,1\n");
  const auto path = std::filesystem::temp_directory_path() / ("thorium2_out_" + std::to_string(::getpid()) + ".txt");
  {
    ResultFile f(path.string(), m);
    f.add(0.0, {300.0, 300.0}, {});
    f.add(10.0, {301.5, 299.25}, {12.0, -3.5});
    f.save();
  }
  std::ifstream in(path, std::ios::binary);
  std::stringstream s;
  s << in.rdbuf();
  EXPECT_EQ(s.str(),
            "Element_number Time,s Temperature,K  Input_heat_power,W\r\n"
            "1 0 300 0\r\n"
            "1 10 301.5 12\r\n"
            "2 0 300 0\r\n"
            "2 10 299.25 -3.5\r\n");
  std::filesystem::remove(path);
}
