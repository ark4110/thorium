// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/output.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

namespace th {

namespace {
std::string fmt(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.10g", v);
  return buf;
}
}  // namespace

std::string v1_number(double v) {
  // Math.Round(v, 6) rounds half to even; nearbyint does the same in the default rounding mode.
  double r = std::nearbyint(v * 1e6) / 1e6;
  if (r == 0) r = 0;  // no "-0"
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.15G", r);
  return buf;
}

ResultFile::ResultFile(std::string path, const Model& model) : path_(std::move(path)) {
  for (const Element& e : model.elements) ids_.push_back(e.id);
  std::ofstream probe(path_);  // fail early, before a long computation
  if (!probe) throw std::runtime_error(path_ + ": cannot write the file");
}

void ResultFile::add(double t, const std::vector<double>& temperature, const std::vector<double>& power) {
  times_.push_back(t);
  T_.push_back(temperature);
  Q_.push_back(power.empty() ? std::vector<double>(temperature.size(), 0.0) : power);
}

void ResultFile::save() const {
  std::ofstream out(path_);
  if (!out) throw std::runtime_error(path_ + ": cannot write the file");
  out << "Element_number Time,s Temperature,K  Input_heat_power,W\r\n";
  for (size_t e = 0; e < ids_.size(); ++e)
    for (size_t k = 0; k < times_.size(); ++k)
      out << ids_[e] << ' ' << v1_number(times_[k]) << ' ' << v1_number(T_[k][e]) << ' ' << v1_number(Q_[k][e]) << "\r\n";
  out.flush();
  if (!out) throw std::runtime_error(path_ + ": write error");
}

BalanceTable::BalanceTable(const std::string& path) : out_(path) {
  if (!out_) throw std::runtime_error(path + ": cannot write the file");
  out_ << "Time,s External_absorbed,W Heat_sources,W Emitted,W Absorbed_of_emitted,W Radiated_to_space,W "
          "Into_fixed,W Stored_rate,W Newton_iterations Step_cuts Bounce_limit_hits\n";
}

void BalanceTable::write(const StepReport& r) {
  const PowerBalance& b = r.balance;
  out_ << fmt(r.t) << ' ' << fmt(b.external) << ' ' << fmt(b.heat_sources) << ' ' << fmt(b.emitted) << ' '
       << fmt(b.absorbed) << ' ' << fmt(b.emitted - b.absorbed) << ' ' << fmt(b.into_fixed) << ' ' << fmt(b.stored_rate)
       << ' ' << r.newton_iterations << ' ' << r.cuts << ' ' << r.bounce_limit_hits << '\n';
  out_.flush();
}

void write_mesh_json(const std::string& path, const Model& m) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error(path + ": cannot write the file");
  std::vector<double> heat(m.elements.size(), 0.0);
  for (const HeatSource& h : m.heat_sources)
    for (int e : h.elements) heat[static_cast<size_t>(e)] += h.power(0, m.integration.t_start) / static_cast<double>(h.elements.size());
  std::vector<bool> fixed(m.elements.size(), false);
  for (const FixedTemperature& f : m.fixed) fixed[static_cast<size_t>(f.element)] = true;
  out << "{\"nodes\":[";
  for (size_t i = 0; i < m.nodes.size(); ++i)
    out << (i ? "," : "") << '[' << fmt(m.nodes[i].x) << ',' << fmt(m.nodes[i].y) << ',' << fmt(m.nodes[i].z) << ']';
  out << "],\"elements\":[";
  for (size_t ei = 0; ei < m.elements.size(); ++ei) {
    const Element& e = m.elements[ei];
    out << (ei ? "," : "") << "{\"id\":" << e.id << ",\"type\":\"" << to_string(e.type) << "\",\"pid\":" << e.pid
        << ",\"faces\":[";
    bool first = true;
    auto polygon = [&](const Face& f) {
      out << (first ? "" : ",") << '[';
      for (size_t k = 0; k < f.nodes.size(); ++k) out << (k ? "," : "") << f.nodes[k];
      out << ']';
      first = false;
    };
    if (e.dim == 2) polygon(m.faces[static_cast<size_t>(e.faces[0])]);
    else
      for (int fi : e.faces)
        if (m.faces[static_cast<size_t>(fi)].radiating) polygon(m.faces[static_cast<size_t>(fi)]);
    out << "],\"medium\":" << m.media[static_cast<size_t>(e.medium)].id << ",\"mass\":" << fmt(e.mass)
        << ",\"volume\":" << fmt(e.volume);
    if (e.dim == 2) out << ",\"area\":" << fmt(e.area) << ",\"thickness\":" << fmt(e.thickness);
    out << ",\"optics\":[";
    for (size_t k = 0; k < e.faces.size(); ++k) {
      const Face& f = m.faces[static_cast<size_t>(e.faces[k])];
      out << (k ? "," : "") << (f.optics >= 0 ? m.optics[static_cast<size_t>(f.optics)].id : 0);
    }
    out << "],\"t0\":" << fmt(m.initial_temperature[ei]) << ",\"heat_source\":" << fmt(heat[ei])
        << ",\"fixed\":" << (fixed[ei] ? "true" : "false") << '}';
  }
  out << "],\"time\":{\"start\":" << fmt(m.integration.t_start) << ",\"end\":" << fmt(m.integration.t_end)
      << ",\"step\":" << fmt(m.integration.dt) << "},\"photons\":" << m.integration.n_photons << ",\"warnings\":[";
  for (size_t i = 0; i < m.warnings.size(); ++i) {
    std::string w;
    for (char c : m.warnings[i]) {
      if (c == '"' || c == '\\') w += '\\';
      if (static_cast<unsigned char>(c) >= 0x20) w += c;
    }
    out << (i ? "," : "") << '"' << w << '"';
  }
  out << "]}\n";
  if (!out) throw std::runtime_error(path + ": write error");
}

int available_cpus() {
  int n = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
#ifdef __linux__
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof set, &set) == 0) n = std::max(1, std::min(n, CPU_COUNT(&set)));
  // cgroup v2 quota: "max 100000" or "150000 100000".
  std::ifstream f("/sys/fs/cgroup/cpu.max");
  std::string quota, period;
  if (f >> quota >> period && quota != "max") {
    const double q = std::stod(quota), p = std::stod(period);
    if (q > 0 && p > 0) n = std::max(1, std::min(n, static_cast<int>(std::ceil(q / p))));
  }
#endif
  return n;
}

}  // namespace th
