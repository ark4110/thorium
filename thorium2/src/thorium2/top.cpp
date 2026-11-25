// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/top.h"

#include <cmath>
#include <set>

namespace th {

namespace {

std::string at(const Card& c) { return c.file + ":" + std::to_string(c.line); }

/// Constant (real literal) or DEPEND number (integer literal).
TopValue value(const Card& c, int idx, std::optional<double> def = std::nullopt) {
  const std::string& s = c.field(idx);
  if (s.empty()) {
    if (def) return TopValue::constant_of(*def);
    c.fail("field " + std::to_string(idx) + " is blank");
  }
  if (auto i = parse_integer(s)) {
    if (*i <= 0) c.fail("field " + std::to_string(idx) + " = '" + s + "': a DEPEND number must be positive");
    return TopValue::dependency_of(*i);
  }
  return TopValue::constant_of(c.real(idx));
}

void require_blank(const Card& c, int from, int to, const std::string& why) {
  for (int i = from; i <= to; ++i)
    if (!c.blank(i)) c.fail("field " + std::to_string(i) + " = '" + c.field(i) + "': " + why);
}

template <class Map, class Item>
void insert_unique(const Card& c, Map& map, int id, Item item) {
  if (id <= 0) c.fail("number must be positive");
  if (!map.emplace(id, std::move(item)).second) c.fail("duplicate number " + std::to_string(id));
}

void parse_ftop(const Card& c, TopModel& m) {
  TopFtop f;
  f.id = c.integer(2);
  f.r_dif = value(c, 3);
  f.r_ref = value(c, 4);
  f.absorb = value(c, 5);
  f.transmit = value(c, 6);
  f.eps_given = !c.blank(7);
  f.eps = f.eps_given ? value(c, 7) : f.absorb;
  if (!c.blank(8) || !c.blank(9))
    c.fail("CDF / T_eff (emission with a given spectrum) is not supported; give a gray emissivity in field 7");
  require_blank(c, 10, 9 + 8 * (c.line_count() - 1), "FTOP has no continuation");
  f.where = at(c);
  insert_unique(c, m.ftop, f.id, f);
}

void parse_medium(const Card& c, TopModel& m) {
  TopMedium md;
  md.id = c.integer(2);
  md.n = value(c, 3, 1.0);
  md.k = value(c, 4, 0.0);
  md.lambda = value(c, 5);
  md.c = value(c, 6);
  require_blank(c, 7, 9, "unexpected field");
  for (int i = 0; i < 6; ++i) md.ftop[i] = c.integer(10 + i, 0);
  require_blank(c, 16, 9 + 8 * (std::max(c.line_count(), 2) - 1), "MEDIUM1 has one continuation line");
  md.where = at(c);
  insert_unique(c, m.media, md.id, md);
}

void parse_depend(const Card& c, TopModel& m) {
  TopDepend d;
  d.id = c.integer(2);
  d.idw_power = c.real(3, 2.0);
  require_blank(c, 4, 9, "unexpected field");
  const int lines = c.line_count() - 1;
  if (lines < 2 || lines % 2 != 0)
    c.fail("each point needs two lines (arguments, then values); found " + std::to_string(lines) + " lines");
  for (int p = 0; p < lines / 2; ++p) {
    const int arg_line = 1 + 2 * p, val_line = arg_line + 1;
    std::array<std::optional<double>, 6> args;
    bool any = false;
    for (int k = 0; k < 6; ++k) {
      const std::string& s = c.field(arg_line, 2 + k);
      if (s.empty()) continue;
      auto v = parse_real(s);
      if (!v) c.fail("point " + std::to_string(p + 1) + ": argument '" + s + "' is not a number");
      args[k] = *v;
      any = true;
    }
    for (int k = 8; k <= 9; ++k)
      if (!c.field(arg_line, k).empty()) c.fail("point " + std::to_string(p + 1) + ": at most 6 argument components");
    if (!any) c.fail("point " + std::to_string(p + 1) + ": argument line is blank");
    std::vector<std::optional<double>> vals(8);
    bool any_value = false;
    for (int k = 0; k < 8; ++k) {
      const std::string& s = c.field(val_line, 2 + k);
      if (s.empty()) continue;
      auto v = parse_real(s);
      if (!v) c.fail("point " + std::to_string(p + 1) + ": value '" + s + "' is not a number");
      vals[k] = *v;
      any_value = true;
    }
    if (!any_value) c.fail("point " + std::to_string(p + 1) + ": value line is blank");
    d.args.push_back(args);
    d.values.push_back(vals);
  }
  d.where = at(c);
  insert_unique(c, m.depend, d.id, d);
}

void parse_rsource(const Card& c, TopModel& m) {
  TopRsource s;
  s.id = c.integer(2);
  if (c.blank(4) && c.blank(5) && is_integer_literal(c.field(3))) {
    s.position_is_dependency = true;
    s.position_dependency = c.integer(3);
    if (s.position_dependency <= 0) c.fail("POS dependency number must be positive");
  } else {
    s.x = value(c, 3);
    s.y = value(c, 4);
    s.z = value(c, 5);
  }
  if (!c.blank(8)) c.fail("CDF (source with a given spectrum) is not supported; give the power Q and leave CDF blank");
  s.power = value(c, 6);
  s.temperature = value(c, 7, 0.0);
  const std::string& nph = c.field(9);
  if (!nph.empty()) {
    const double v = c.real(9);
    if (v < 0 || v != std::floor(v)) c.fail("N_PH must be a non-negative whole number");
    s.n_photons = static_cast<int>(v);
  }
  require_blank(c, 10, 9 + 8 * (c.line_count() - 1), "RSOURCE1 has no continuation");
  s.where = at(c);
  insert_unique(c, m.rsources, s.id, s);
}

void parse_hsource(const Card& c, TopModel& m) {
  TopHsource h;
  h.id = c.integer(2);
  h.power = value(c, 3);
  require_blank(c, 4, 9, "elements are listed on continuation lines");
  for (int i = 10; i < 2 + 8 * c.line_count(); ++i)
    if (!c.blank(i)) h.elements.push_back(c.integer(i));
  if (h.elements.empty()) c.fail("no elements listed");
  h.where = at(c);
  insert_unique(c, m.hsources, h.id, h);
}

template <class Pair, class Read>
std::vector<Pair> read_pairs(const Card& c, Read read) {
  std::vector<Pair> out;
  require_blank(c, 3, 9, "pairs are listed on continuation lines");
  for (int line = 1; line < c.line_count(); ++line) {
    for (int f = 2; f <= 8; f += 2) {
      const bool a = c.field(line, f).empty(), b = c.field(line, f + 1).empty();
      if (a && b) continue;
      const int idx = 2 + 8 * line + (f - 2);
      if (a || b) c.fail("line " + std::to_string(line + 1) + ": element and temperature must be given in pairs");
      out.push_back({c.integer(idx), read(idx + 1)});
    }
  }
  if (out.empty()) c.fail("no elements listed");
  return out;
}

void parse_iparam(const Card& c, TopModel& m) {
  if (m.iparam) c.fail("I_PARAM is given twice");
  TopIparam p;
  p.type = c.field(2);
  static const std::set<std::string> types = {"EULER", "EULER_V", "ADAMS_4", "BDF2", "BE"};
  if (!types.count(p.type)) c.fail("integrator '" + p.type + "' is unknown; use BDF2, BE (or EULER, EULER_V, ADAMS_4 of version 1)");
  p.t_start = c.real(3);
  p.t_length = c.real(4);
  p.t_step = c.real(5);
  if (!(p.t_length > 0)) c.fail("t_LENGTH must be positive");
  if (!(p.t_step > 0)) c.fail("t_STEP must be positive");
  const double nph = c.real(6);
  if (!(nph >= 1) || nph != std::floor(nph) || nph > 2e9) c.fail("N_PH must be a positive whole number");
  p.n_photons = static_cast<int>(nph);
  if (!c.blank(7)) p.kt_min = c.real(7);
  if (!c.blank(8)) p.kt_max = c.real(8);
  p.smooth = c.field(10);
  p.smooth_steps = c.integer(11, 0);
  p.where = at(c);
  m.iparam = p;
}

}  // namespace

TopModel parse_top(const std::vector<Card>& cards) {
  TopModel m;
  for (const Card& c : cards) {
    if (c.name == "FTOP") parse_ftop(c, m);
    else if (c.name == "MEDIUM1") parse_medium(c, m);
    else if (c.name == "DEPEND") parse_depend(c, m);
    else if (c.name == "RSOURCE1") parse_rsource(c, m);
    else if (c.name == "HSOURCE1") parse_hsource(c, m);
    else if (c.name == "T_ALL") {
      if (m.t_all) c.fail("T_ALL is given twice");
      m.t_all = c.real(3);
      if (!(*m.t_all > 0)) c.fail("temperature must be positive");
    } else if (c.name == "T_INIT") {
      TopTinit t;
      t.id = c.integer(2);
      t.temperatures = read_pairs<std::pair<int, double>>(c, [&](int i) {
        const double v = c.real(i);
        if (!(v > 0)) c.fail("temperature must be positive");
        return v;
      });
      t.where = at(c);
      m.t_init.push_back(std::move(t));
    } else if (c.name == "T_SET") {
      TopTset t;
      t.id = c.integer(2);
      t.temperatures = read_pairs<std::pair<int, TopValue>>(c, [&](int i) { return value(c, i); });
      t.where = at(c);
      m.t_set.push_back(std::move(t));
    } else if (c.name == "G_MEDIUM") {
      if (m.g_medium) c.fail("G_MEDIUM is given twice");
      m.g_medium = c.integer(2);
    } else if (c.name == "I_PARAM") parse_iparam(c, m);
    else c.fail("unknown card (supported: FTOP, MEDIUM1, DEPEND, RSOURCE1, HSOURCE1, T_ALL, T_INIT, T_SET, G_MEDIUM, I_PARAM)");
  }
  return m;
}

TopModel read_top_file(const std::string& path) { return parse_top(read_cards_file(path)); }

}  // namespace th
