#include "reference.h"

#include "thorium2/bdf.h"
#include "thorium2/top.h"

#include <cmath>
#include <cstdio>
#include <sstream>

namespace ref {

th::Model model_from_text(const std::string& bdf, const std::string& top) {
  std::istringstream b(bdf), t(top);
  return th::build_model(th::parse_bdf(th::read_cards(b, "test.bdf")), th::parse_top(th::read_cards(t, "test.top")));
}

std::string num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  std::string s = buf;
  if (s.find_first_of(".eEn") == std::string::npos) s += ".";
  return s;
}

std::string card(const std::string& name, const std::vector<std::string>& fields) {
  // Free field: name + 8 data fields per line; continuation lines start with ','.
  std::string out = name;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i > 0 && i % 8 == 0) out += "\n";
    out += "," + fields[i];
  }
  return out + "\n";
}

std::string plate(int nx, int ny, double lx, double ly, double z0, int pid, int grid0, int elem0, bool flip) {
  std::string s;
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i <= nx; ++i)
      s += card("GRID", {std::to_string(grid0 + 1 + j * (nx + 1) + i), "", num(lx * i / nx), num(ly * j / ny), num(z0)});
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const int g1 = grid0 + 1 + j * (nx + 1) + i, g2 = g1 + 1, g3 = g2 + nx + 1, g4 = g1 + nx + 1;
      std::vector<std::string> f = {std::to_string(elem0 + 1 + j * nx + i), std::to_string(pid)};
      if (flip) f.insert(f.end(), {std::to_string(g4), std::to_string(g3), std::to_string(g2), std::to_string(g1)});
      else f.insert(f.end(), {std::to_string(g1), std::to_string(g2), std::to_string(g3), std::to_string(g4)});
      s += card("CQUAD4", f);
    }
  return s;
}

std::string hex_rod(int n, double len, double side, int pid, int grid0, int elem0) {
  std::string s;
  // 4 grids per station: (y,z) = (0,0), (s,0), (s,s), (0,s)
  const double yz[4][2] = {{0, 0}, {side, 0}, {side, side}, {0, side}};
  for (int i = 0; i <= n; ++i)
    for (int k = 0; k < 4; ++k)
      s += card("GRID", {std::to_string(grid0 + 1 + 4 * i + k), "", num(len * i / n), num(yz[k][0]), num(yz[k][1])});
  for (int i = 0; i < n; ++i) {
    // Bottom face G1..G4 at station i must be counter-clockwise seen from G5..G8 (station i+1, +x).
    const int a = grid0 + 1 + 4 * i, b = a + 4;
    std::vector<std::string> f = {std::to_string(elem0 + 1 + i), std::to_string(pid)};
    for (int k : {0, 3, 2, 1}) f.push_back(std::to_string(a + k));
    for (int k : {0, 3, 2, 1}) f.push_back(std::to_string(b + k));
    s += card("CHEXA", f);
  }
  return s;
}

std::string top_header(double t_length, double dt, int n_photons, double t_all) {
  return card("FTOP", {"1", "0.", "0.", "1.", "0.", "1."}) +
         card("I_PARAM", {"BDF2", "0.", num(t_length), num(dt), std::to_string(n_photons)}) +
         card("T_ALL", {"1", num(t_all)});
}

double F_parallel_rectangles(double a, double b, double c) {
  const double X = a / c, Y = b / c;
  const double t1 = std::log(std::sqrt((1 + X * X) * (1 + Y * Y) / (1 + X * X + Y * Y)));
  const double t2 = X * std::sqrt(1 + Y * Y) * std::atan(X / std::sqrt(1 + Y * Y));
  const double t3 = Y * std::sqrt(1 + X * X) * std::atan(Y / std::sqrt(1 + X * X));
  return 2.0 / (M_PI * X * Y) * (t1 + t2 + t3 - X * std::atan(X) - Y * std::atan(Y));
}

double F_perpendicular_rectangles(double h, double w, double l) {
  const double H = h / l, W = w / l, H2 = H * H, W2 = W * W, S = std::sqrt(H2 + W2);
  const double a = (1 + W2) * (1 + H2) / (1 + W2 + H2);
  const double b = W2 * (1 + W2 + H2) / ((1 + W2) * (W2 + H2));
  const double c = H2 * (1 + H2 + W2) / ((1 + H2) * (H2 + W2));
  const double log_term = 0.25 * (std::log(a) + W2 * std::log(b) + H2 * std::log(c));
  return 1.0 / (M_PI * W) * (W * std::atan(1 / W) + H * std::atan(1 / H) - S * std::atan(1 / S) + log_term);
}

std::vector<double> rk4(const std::function<std::vector<double>(double, const std::vector<double>&)>& f,
                        std::vector<double> y, double t0, double t_end, int steps) {
  const double h = (t_end - t0) / steps;
  auto axpy = [](const std::vector<double>& a, const std::vector<double>& b, double s) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + s * b[i];
    return r;
  };
  double t = t0;
  for (int i = 0; i < steps; ++i) {
    auto k1 = f(t, y);
    auto k2 = f(t + h / 2, axpy(y, k1, h / 2));
    auto k3 = f(t + h / 2, axpy(y, k2, h / 2));
    auto k4 = f(t + h, axpy(y, k3, h));
    for (size_t j = 0; j < y.size(); ++j) y[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
    t += h;
  }
  return y;
}

}  // namespace ref
