// The command line program, run as a separate process.
#include "reference.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace fs = std::filesystem;
using ref::card;

namespace {

struct Result {
  int code;
  std::string output;  // stdout + stderr
};

Result run(const std::string& args) {
  const std::string cmd = std::string(THORIUM2_CLI) + " " + args + " 2>&1";
  FILE* p = popen(cmd.c_str(), "r");
  std::string out;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
  const int status = pclose(p);
  return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, out};
}

std::string read(const fs::path& p) {
  std::ifstream f(p);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

class Cli : public ::testing::Test {
protected:
  void SetUp() override {
    dir = fs::temp_directory_path() / ("thorium2_cli_" + std::to_string(::getpid()) + "_" +
                                       ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream(dir / "plate.bdf") << ref::plate(2, 1, 2.0, 1.0, 0.0, 10, 0, 0) << card("PSHELL", {"10", "1", ".01"})
                                     << card("MAT1", {"1", "", "", "", "1000."});
    std::ofstream(dir / "plate.top") << card("FTOP", {"1", "1.", "0.", "0.", "0.", "0."})
                                     << card("MEDIUM1", {"10", "1.", "0.", "100.", "500."}) << ",1,1\n"
                                     << card("I_PARAM", {"BDF2", "0.", "100.", "10.", "100"})
                                     << card("T_ALL", {"1", "300."}) << card("HSOURCE1", {"1", "50."}) << ",1\n";
  }
  void TearDown() override { fs::remove_all(dir); }
  fs::path dir;
};

}  // namespace

/// Lines "element time T Q" of a result file.
struct Row {
  int element;
  double t, T, Q;
};
std::vector<Row> rows(const std::string& text) {
  std::vector<Row> out;
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    Row r{};
    if (std::sscanf(line.c_str(), "%d %lf %lf %lf", &r.element, &r.t, &r.T, &r.Q) == 4) out.push_back(r);
  }
  return out;
}

TEST_F(Cli, WritesResultsInVersion1Format) {
  Result r = run((dir / "plate.bdf").string() + " --threads 1 --balance " + (dir / "b.txt").string());
  ASSERT_EQ(r.code, 0) << r.output;
  const std::string text = read(dir / "plate.txt");
  EXPECT_EQ(text.rfind("Element_number Time,s Temperature,K  Input_heat_power,W\r\n1 0 300 0\r\n", 0), 0u) << text;
  const auto rs = rows(text);
  ASSERT_EQ(rs.size(), 2u * 11u);  // 2 elements x (t = 0 and 10 steps), grouped by element
  for (size_t k = 0; k < 11; ++k) {
    EXPECT_EQ(rs[k].element, 1);
    EXPECT_EQ(rs[11 + k].element, 2);
    EXPECT_DOUBLE_EQ(rs[k].t, 10.0 * static_cast<double>(k));
  }
  // Energy: 50 W for 100 s into 2 x 5000 J/K; the heat powers of the two elements add up to 50 W.
  EXPECT_NEAR(rs[10].T + rs[21].T, 601.0, 1e-5);
  EXPECT_GT(rs[10].T, rs[21].T);
  EXPECT_NEAR(rs[10].Q + rs[21].Q, 50.0, 1e-5);
  EXPECT_NE(read(dir / "b.txt").find("Time,s External_absorbed,W"), std::string::npos);
}

TEST_F(Cli, EveryAndEndOptions) {
  Result r = run((dir / "plate.bdf").string() + " --threads 1 --every 3 --end 50 --out " + (dir / "t.txt").string());
  ASSERT_EQ(r.code, 0) << r.output;
  std::vector<double> times;
  for (const Row& row : rows(read(dir / "t.txt")))
    if (row.element == 1) times.push_back(row.t);
  EXPECT_EQ(times, (std::vector<double>{0.0, 30.0, 50.0}));
}

TEST_F(Cli, MeshExportForViewer) {
  Result r = run((dir / "plate.bdf").string() + " --check --mesh " + (dir / "m.json").string());
  ASSERT_EQ(r.code, 0) << r.output;
  const std::string j = read(dir / "m.json");
  // Nodes, two shells with their polygon, properties and the time span of I_PARAM.
  EXPECT_NE(j.find("\"nodes\":[[0,0,0],[1,0,0],[2,0,0]"), std::string::npos) << j;
  EXPECT_NE(j.find("{\"id\":1,\"type\":\"CQUAD4\",\"pid\":10,\"faces\":[[0,1,4,3]]"), std::string::npos) << j;
  EXPECT_NE(j.find("\"mass\":10"), std::string::npos) << j;
  EXPECT_NE(j.find("\"thickness\":0.01"), std::string::npos) << j;
  EXPECT_NE(j.find("\"t0\":300"), std::string::npos) << j;
  EXPECT_NE(j.find("\"time\":{\"start\":0,\"end\":100,\"step\":10}"), std::string::npos) << j;
  EXPECT_NE(j.find("\"heat_source\":50"), std::string::npos) << j;
}

TEST_F(Cli, CheckOnlyPrintsSummary) {
  Result r = run((dir / "plate.bdf").string() + " --check");
  ASSERT_EQ(r.code, 0) << r.output;
  EXPECT_NE(r.output.find("2 elements"), std::string::npos) << r.output;
  EXPECT_FALSE(fs::exists(dir / "plate.txt"));
}

TEST_F(Cli, InputErrorsExitWithCodeOneAndPlace) {
  std::ofstream(dir / "bad.bdf") << "GRID    1               0.      0.      0.\nCTRIA3  1       10      1       2       3\n";
  std::ofstream(dir / "bad.top") << "FTOP    1       1.      0.      0.      0.\n";
  Result r = run((dir / "bad.bdf").string());
  EXPECT_EQ(r.code, 1);
  EXPECT_NE(r.output.find("I_PARAM"), std::string::npos) << r.output;

  std::ofstream(dir / "bad.top") << "FTOP    1       1.      0.      0.      0.\nI_PARAM BDF2    0.      1.      1.      10\n"
                                 << "T_ALL   1       300.\n";
  r = run((dir / "bad.bdf").string());
  EXPECT_EQ(r.code, 1);
  EXPECT_NE(r.output.find("bad.bdf:2"), std::string::npos) << r.output;
  EXPECT_NE(r.output.find("grid 2"), std::string::npos) << r.output;

  r = run((dir / "missing.bdf").string());
  EXPECT_EQ(r.code, 1);
  r = run("--unknown-option x.bdf");
  EXPECT_EQ(r.code, 1);
}

TEST_F(Cli, Help) {
  Result r = run("--help");
  EXPECT_EQ(r.code, 0);
  EXPECT_NE(r.output.find("--top"), std::string::npos);
  EXPECT_NE(run("--version").output.find("2."), std::string::npos);
}
