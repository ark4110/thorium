// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/table.h"

#include <algorithm>
#include <cmath>

namespace th {

namespace {
const char* kArgNames[6] = {"temperature", "time", "random number", "radiation frequency", "elevation angle",
                            "azimuth angle"};

size_t index_of(const std::vector<double>& axis, double v) {
  return static_cast<size_t>(std::lower_bound(axis.begin(), axis.end(), v) - axis.begin());
}

/// Interval index i and weight w so that x ~ (1-w) xs[i] + w xs[i+1]; clamped at the ends.
void locate(const std::vector<double>& xs, double x, size_t& i, double& w) {
  if (xs.size() < 2 || x <= xs.front()) {
    i = 0;
    w = 0;
    return;
  }
  if (x >= xs.back()) {
    i = xs.size() - 2;
    w = 1;
    return;
  }
  i = static_cast<size_t>(std::upper_bound(xs.begin(), xs.end(), x) - xs.begin()) - 1;
  w = (x - xs[i]) / (xs[i + 1] - xs[i]);
}
}  // namespace

Table::Table(const TopDepend& d) : id_(d.id), where_(d.where) {
  auto fail = [&](const std::string& msg) -> void {
    throw InputError(d.where + ": DEPEND " + std::to_string(d.id) + ": " + msg);
  };
  const size_t n = d.args.size();
  if (n == 0) fail("no points");

  // Value components: contiguous from the first, same count in every point.
  for (size_t p = 0; p < n; ++p) {
    int last = -1;
    for (int k = 0; k < 8; ++k)
      if (d.values[p][k]) last = k;
    for (int k = 0; k <= last; ++k)
      if (!d.values[p][k]) fail("point " + std::to_string(p + 1) + ": value component " + std::to_string(k + 1) + " is blank");
    if (p == 0) n_values_ = last + 1;
    else if (last + 1 != n_values_) fail("points have different numbers of value components");
  }

  // Active argument components.
  bool active[6] = {};
  for (int k = 0; k < 6; ++k) {
    size_t given = 0;
    for (size_t p = 0; p < n; ++p) given += d.args[p][k].has_value();
    if (given == 0) continue;
    if (given != n) fail(std::string("argument '") + kArgNames[k] + "' is given in some points and blank in others");
    for (size_t p = 1; p < n; ++p)
      if (*d.args[p][k] != *d.args[0][k]) active[k] = true;
  }
  for (int k = 2; k < 6; ++k)
    if (active[k]) fail(std::string("dependency on ") + kArgNames[k] + " is not supported (only temperature and time)");

  auto collect = [&](int k, std::vector<double>& axis) {
    for (size_t p = 0; p < n; ++p) axis.push_back(*d.args[p][k]);
    std::sort(axis.begin(), axis.end());
    axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
  };
  if (active[kArgTemperature]) collect(kArgTemperature, t_points_);
  if (active[kArgTime]) collect(kArgTime, time_points_);
  t_axis_ = active[kArgTemperature] ? 0 : -1;
  time_axis_ = active[kArgTime] ? 1 : -1;

  const size_t nT = std::max<size_t>(1, t_points_.size());
  const size_t nt = std::max<size_t>(1, time_points_.size());
  if (!active[kArgTemperature] && !active[kArgTime]) {
    // Constant: all points must agree.
    for (size_t p = 1; p < n; ++p)
      for (int k = 0; k < n_values_; ++k)
        if (*d.values[p][k] != *d.values[0][k]) fail("points have the same arguments but different values");
  } else if (nT * nt != n) {
    fail(active[kArgTemperature] && active[kArgTime]
             ? "points of a temperature-time table must form a rectangular grid (every temperature at every time)"
             : "the same argument value is given twice");
  }
  values_.assign(static_cast<size_t>(n_values_), std::vector<double>(nT * nt, 0.0));
  std::vector<bool> filled(nT * nt, false);
  for (size_t p = 0; p < n; ++p) {
    const size_t iT = active[kArgTemperature] ? index_of(t_points_, *d.args[p][kArgTemperature]) : 0;
    const size_t it = active[kArgTime] ? index_of(time_points_, *d.args[p][kArgTime]) : 0;
    const size_t cell = it * nT + iT;
    if (filled[cell] && (active[kArgTemperature] || active[kArgTime]))
      fail("the same argument value is given twice (point " + std::to_string(p + 1) + ")");
    filled[cell] = true;
    for (int k = 0; k < n_values_; ++k) values_[static_cast<size_t>(k)][cell] = *d.values[p][k];
  }
}

double Table::eval(double T, double t, int component) const {
  if (component < 0 || component >= n_values_)
    throw InputError(where_ + ": DEPEND " + std::to_string(id_) + ": value component " + std::to_string(component + 1) +
                     " is required but the table has " + std::to_string(n_values_));
  const auto& v = values_[static_cast<size_t>(component)];
  const size_t nT = std::max<size_t>(1, t_points_.size());
  size_t iT = 0, it = 0;
  double wT = 0, wt = 0;
  if (t_axis_ >= 0) locate(t_points_, T, iT, wT);
  if (time_axis_ >= 0) locate(time_points_, t, it, wt);
  const size_t iT1 = (t_axis_ >= 0 && t_points_.size() > 1) ? iT + 1 : iT;
  const size_t it1 = (time_axis_ >= 0 && time_points_.size() > 1) ? it + 1 : it;
  const double a = (1 - wT) * v[it * nT + iT] + wT * v[it * nT + iT1];
  const double b = (1 - wT) * v[it1 * nT + iT] + wT * v[it1 * nT + iT1];
  return (1 - wt) * a + wt * b;
}

double Table::eval_1d(const std::vector<double>& xs, const std::vector<double>& ys, double x) const {
  size_t i = 0;
  double w = 0;
  locate(xs, x, i, w);
  return xs.size() < 2 ? ys[0] : (1 - w) * ys[i] + w * ys[i + 1];
}

double Table::integral_1d(const std::vector<double>& xs, const std::vector<double>& ys, double a, double b) const {
  // F(x) = integral from xs[0] to x of the clamped piecewise-linear interpolant.
  auto F = [&](double x) {
    if (x <= xs.front()) return ys.front() * (x - xs.front());
    double s = 0;
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
      if (x <= xs[i + 1]) {
        const double yx = eval_1d(xs, ys, x);
        return s + 0.5 * (ys[i] + yx) * (x - xs[i]);
      }
      s += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]);
    }
    return s + ys.back() * (x - xs.back());
  };
  return F(b) - F(a);
}

double Table::integrate_temperature(double T0, double T1, double t) const {
  if (t_axis_ < 0) return eval(0.0, t, 0) * (T1 - T0);
  std::vector<double> column(t_points_.size());
  for (size_t i = 0; i < t_points_.size(); ++i) column[i] = eval(t_points_[i], t, 0);
  return integral_1d(t_points_, column, T0, T1);
}

}  // namespace th
