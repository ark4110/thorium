// Thorium 2 — command line program.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/model.h"
#include "thorium2/output.h"
#include "thorium2/solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <type_traits>
#include <string>

#ifdef THORIUM2_HAVE_OPENMP
#include <omp.h>
#endif

namespace {

constexpr const char* kVersion = "2.0.0";

const char* kHelp =
    "Thorium 2 - transient radiation-conduction heat transfer by finite elements and Monte Carlo.\n"
    "\n"
    "Usage: thorium2 [options] MODEL.bdf\n"
    "\n"
    "The *.top file (properties, loads, I_PARAM) has the same name as the model unless --top is given.\n"
    "\n"
    "Options:\n"
    "  --top FILE       *.top file\n"
    "  --out FILE       results in the text format of version 1 (default: MODEL.txt):\n"
    "                   Element_number Time,s Temperature,K Input_heat_power,W\n"
    "  --balance FILE   power balance of the model and solver statistics per step (text)\n"
    "  --every N        write every N-th step (the first and the last are always written)\n"
    "  --end T          stop at time T instead of t_START + t_LENGTH of I_PARAM\n"
    "  --steps N        stop after N steps\n"
    "  --photons N      photon bundles per face and step (default: N_PH of I_PARAM)\n"
    "  --seed S         random seed (default 1); the same seed gives the same result\n"
    "  --threads N      threads for ray tracing (default: CPUs available to the process)\n"
    "  --check          read and check the model, print a summary, do not integrate\n"
    "  --mesh FILE      write geometry and properties of the model for a viewer (JSON)\n"
    "  --quiet          no progress output\n"
    "  --help, --version\n"
    "\n"
    "Exit status: 0 success, 1 error in the input or the command line, 2 failure of the computation.\n";

struct Options {
  std::string model, top, out, balance, mesh;
  long every = 1;
  std::optional<double> end;
  std::optional<long> steps;
  int photons = 0;
  unsigned long long seed = 1;
  int threads = 0;
  bool check = false, quiet = false;
};

[[noreturn]] void usage_error(const std::string& msg) {
  std::cerr << "thorium2: " << msg << "\n(see thorium2 --help)\n";
  std::exit(1);
}

template <class T>
T number(const std::string& opt, const std::string& v) {
  try {
    size_t pos = 0;
    T x{};
    if constexpr (std::is_same_v<T, double>) x = std::stod(v, &pos);
    else if constexpr (std::is_same_v<T, unsigned long long>) x = std::stoull(v, &pos);
    else x = static_cast<T>(std::stol(v, &pos));
    if (pos != v.size()) throw std::invalid_argument(v);
    return x;
  } catch (const std::exception&) {
    usage_error(opt + ": '" + v + "' is not a valid number");
  }
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      std::cout << kHelp;
      std::exit(0);
    } else if (a == "--version") {
      std::cout << "thorium2 " << kVersion << "\n";
      std::exit(0);
    } else if (a == "--top") o.top = value();
    else if (a == "--out") o.out = value();
    else if (a == "--balance") o.balance = value();
    else if (a == "--mesh") o.mesh = value();
    else if (a == "--every") o.every = number<long>(a, value());
    else if (a == "--end") o.end = number<double>(a, value());
    else if (a == "--steps") o.steps = number<long>(a, value());
    else if (a == "--photons") o.photons = number<int>(a, value());
    else if (a == "--seed") o.seed = number<unsigned long long>(a, value());
    else if (a == "--threads") o.threads = number<int>(a, value());
    else if (a == "--check") o.check = true;
    else if (a == "--quiet") o.quiet = true;
    else if (!a.empty() && a[0] == '-') usage_error("unknown option " + a);
    else if (o.model.empty()) o.model = a;
    else usage_error("more than one model given: " + o.model + ", " + a);
  }
  if (o.model.empty()) usage_error("no model given");
  if (o.every < 1) usage_error("--every must be at least 1");
  if (o.steps && *o.steps < 1) usage_error("--steps must be at least 1");
  if (o.photons < 0) usage_error("--photons must be positive");
  if (o.threads < 0) usage_error("--threads must be positive");
  if (o.out.empty()) {
    std::filesystem::path p(o.model);
    p.replace_extension(".txt");
    o.out = p.string();
  }
  return o;
}

void summary(const th::Model& m) {
  int shells = 0, solids = 0, radiating = 0;
  double mass = 0;
  for (const auto& e : m.elements) (e.dim == 2 ? shells : solids)++, mass += e.mass;
  for (const auto& f : m.faces) radiating += f.radiating;
  std::printf("Model: %zu elements (%d shells, %d solids), %zu grids, mass %.6g kg\n", m.elements.size(), shells, solids,
              m.nodes.size(), mass);
  std::printf("       %d radiating faces, %zu conduction contacts, %zu radiation sources, %zu heat sources, %zu fixed\n",
              radiating, m.junctions.size(), m.sources.size(), m.heat_sources.size(), m.fixed.size());
  std::printf("Time:  %.6g .. %.6g s, step %.6g s, %d photon bundles per face and step\n", m.integration.t_start,
              m.integration.t_end, m.integration.dt, m.integration.n_photons);
  for (const auto& w : m.warnings) std::printf("warning: %s\n", w.c_str());
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  th::Model model;
  try {
    model = th::load_model(o.model, o.top);
  } catch (const th::InputError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  if (o.end) {
    if (!(*o.end > model.integration.t_start)) {
      std::cerr << "error: --end must be after t_START = " << model.integration.t_start << "\n";
      return 1;
    }
    model.integration.t_end = *o.end;
  }
  if (!o.mesh.empty()) {
    try {
      th::write_mesh_json(o.mesh, model);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
    }
  }
  if (!o.quiet || o.check) summary(model);
  if (o.check) return 0;

  const int threads = o.threads > 0 ? o.threads : th::available_cpus();
#ifdef THORIUM2_HAVE_OPENMP
  omp_set_num_threads(threads);
#endif

  std::optional<th::ResultFile> results;
  try {
    th::SolverSettings settings;
    settings.seed = o.seed;
    settings.photons_per_face = o.photons;
    th::Solver solver(model, settings);
    results.emplace(o.out, model);
    std::optional<th::BalanceTable> balance;
    if (!o.balance.empty()) balance.emplace(o.balance);
    results->add(solver.time(), solver.temperatures(), {});

    const auto start = std::chrono::steady_clock::now();
    auto last_progress = start;
    long n = 0;
    while (!solver.finished() && (!o.steps || n < *o.steps)) {
      const th::StepReport r = solver.step();
      ++n;
      const bool last = solver.finished() || (o.steps && n == *o.steps);
      if (n % o.every == 0 || last) results->add(r.t, solver.temperatures(), solver.heat_flow());
      if (balance) balance->write(r);
      if (r.bounce_limit_hits > 0)
        std::fprintf(stderr, "warning: t = %g s: %lld photon bundles stopped after 10000 interactions\n", r.t, r.bounce_limit_hits);
      const auto now = std::chrono::steady_clock::now();
      if (!o.quiet && (now - last_progress > std::chrono::seconds(2) || last)) {
        const auto& T = solver.temperatures();
        const auto [lo, hi] = std::minmax_element(T.begin(), T.end());
        std::fprintf(stderr, "t = %-12.6g step %-8ld T = %.2f .. %.2f K  newton %d%s\n", r.t, n, *lo, *hi,
                     r.newton_iterations, r.cuts ? "  (step cut)" : "");
        last_progress = now;
      }
    }
    results->save();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!o.quiet)
      std::printf("Done: %ld steps in %.3g s (%d threads). Results: %s\n", n, seconds, threads, results->path().c_str());
  } catch (const th::InputError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    // Keep what was computed up to the failure.
    if (results) {
      try {
        results->save();
        std::cerr << "results up to the failure are in " << results->path() << "\n";
      } catch (const std::exception& e2) {
        std::cerr << "error: " << e2.what() << "\n";
      }
    }
    return 2;
  }
  return 0;
}
