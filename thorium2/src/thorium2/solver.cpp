// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/solver.h"

#include "thorium2/conduction.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace th {

struct Solver::Impl {
  std::vector<int> unknown;  ///< Element -> unknown index, -1 for T_SET elements.
  std::vector<int> elements; ///< Unknown index -> element.
  std::vector<int> fixed_of; ///< Element -> index into Model::fixed, -1 if free.
  std::vector<int> emitters; ///< Radiating faces.
};

namespace {

/// Data that stays constant during one (sub)step.
struct StepData {
  const RadiationResult* rad = nullptr;
  double t1 = 0, dt = 0;
  double a0 = 1, a1 = -1, a2 = 0;
  std::vector<double> history;  ///< a1 m H(T_n) + a2 m H(T_{n-1}) per element.
  std::vector<double> sources;  ///< External radiation + heat sources, W per element.
};

struct Flows {
  std::vector<double> q;        ///< Net heat flow into each element, W.
  std::vector<Triplet> dq;      ///< dq_row / dT_col.
  double emitted = 0, absorbed = 0;
};

}  // namespace

Solver::Solver(const Model& model, SolverSettings settings)
    : m_(model), s_(settings), t_(model.integration.t_start), T_(model.initial_temperature),
      q_(model.elements.size(), 0.0), impl_(std::make_unique<Impl>()) {
  const size_t n = m_.elements.size();
  impl_->unknown.assign(n, 0);
  impl_->fixed_of.assign(n, -1);
  for (size_t k = 0; k < m_.fixed.size(); ++k) impl_->fixed_of[static_cast<size_t>(m_.fixed[k].element)] = static_cast<int>(k);
  for (size_t e = 0; e < n; ++e) {
    if (impl_->fixed_of[e] >= 0) {
      impl_->unknown[e] = -1;
    } else {
      impl_->unknown[e] = static_cast<int>(impl_->elements.size());
      impl_->elements.push_back(static_cast<int>(e));
    }
  }
  for (size_t f = 0; f < m_.faces.size(); ++f)
    if (m_.faces[f].radiating) impl_->emitters.push_back(static_cast<int>(f));
  if (!impl_->emitters.empty()) scene_ = std::make_unique<RadiationScene>(m_);
}

Solver::~Solver() = default;

bool Solver::finished() const { return t_ >= m_.integration.t_end - 1e-9 * m_.integration.dt; }

namespace {

Flows compute_flows(const Model& m, const std::vector<int>& emitters, const std::vector<double>& T, const StepData& d,
                    bool with_jacobian) {
  Flows f;
  f.q = d.sources;
  const std::vector<double> lambda = element_conductivity(m, T, d.t1);
  add_conduction(m, T, lambda, f.q, with_jacobian ? &f.dq : nullptr);
  if (!d.rad) return f;
  for (int fi : emitters) {
    const Face& face = m.faces[static_cast<size_t>(fi)];
    const int e = face.element;
    const double Te = T[static_cast<size_t>(e)];
    const double eps = m.optics[static_cast<size_t>(face.optics)].eps(Te, d.t1);
    if (eps == 0) continue;
    const double k = eps * kStefanBoltzmann * face.area;
    const double P = k * Te * Te * Te * Te, dP = 4 * k * Te * Te * Te;
    f.q[static_cast<size_t>(e)] -= P;
    f.emitted += P;
    if (with_jacobian) f.dq.push_back({e, e, -dP});
    for (const auto& [target, F] : d.rad->absorbed[static_cast<size_t>(fi)]) {
      f.q[static_cast<size_t>(target)] += F * P;
      f.absorbed += F * P;
      if (with_jacobian) f.dq.push_back({target, e, F * dP});
    }
  }
  return f;
}

}  // namespace

StepReport Solver::step() {
  if (finished()) throw std::logic_error("Solver::step called after the end of the integration interval");
  const double dt_model = m_.integration.dt;
  double dt = std::min(dt_model, m_.integration.t_end - t_);
  if (m_.integration.t_end - (t_ + dt) < 1e-9 * dt_model) dt = m_.integration.t_end - t_;
  const double t1 = t_ + dt;

  StepReport report;
  RadiationResult rad;
  const RadiationResult* rad_ptr = nullptr;
  if (scene_) {
    RadiationSettings rs;
    rs.photons_per_face = s_.photons_per_face > 0 ? s_.photons_per_face : m_.integration.n_photons;
    rs.seed = s_.seed;
    rad = sample_radiation(m_, *scene_, T_, t1, step_, rs);
    rad_ptr = &rad;
    report.bounce_limit_hits = rad.bounce_limit_hits;
  }
  // Heat sources at the end of the step (backward Euler / BDF2 take the right-hand side there).
  auto sources_at = [&](double t) {
    std::vector<double> s(m_.elements.size(), 0.0);
    if (rad_ptr) s = rad.external;
    for (const HeatSource& h : m_.heat_sources) {
      const double per = h.power(0, t) / static_cast<double>(h.elements.size());
      for (int e : h.elements) s[static_cast<size_t>(e)] += per;
    }
    return s;
  };
  auto fixed_at = [&](std::vector<double>& T, double t) {
    for (const FixedTemperature& f : m_.fixed) T[static_cast<size_t>(f.element)] = f.temperature(0, t);
  };

  const size_t n = m_.elements.size();
  const size_t nu = impl_->elements.size();
  auto enthalpy = [&](size_t e, double T) {
    const Element& el = m_.elements[e];
    return el.mass * m_.media[static_cast<size_t>(el.medium)].enthalpy(T);
  };

  // Solves one (sub)step from T0 over [ta, ta + h]. Returns false if Newton fails.
  auto solve = [&](const std::vector<double>& T0, const std::vector<double>* Tm1, double ta, double h,
                   std::vector<double>& T, int& iterations, Flows& last) -> bool {
    StepData d;
    d.rad = rad_ptr;
    d.t1 = ta + h;
    d.dt = h;
    if (Tm1) d.a0 = 1.5, d.a1 = -2.0, d.a2 = 0.5;
    d.history.assign(n, 0.0);
    for (size_t e = 0; e < n; ++e) {
      if (impl_->unknown[e] < 0) continue;
      d.history[e] = d.a1 * enthalpy(e, T0[e]) + (Tm1 ? d.a2 * enthalpy(e, (*Tm1)[e]) : 0.0);
    }
    d.sources = sources_at(d.t1);
    T = T0;
    fixed_at(T, d.t1);
    if (nu == 0) {
      last = compute_flows(m_, impl_->emitters, T, d, false);
      return true;
    }
    double Tmax = 0;
    for (double x : T) Tmax = std::max(Tmax, x);
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> lu;
    for (int it = 0; it < s_.max_newton; ++it) {
      iterations = it + 1;
      Flows fl = compute_flows(m_, impl_->emitters, T, d, true);
      Eigen::VectorXd R(static_cast<Eigen::Index>(nu));
      std::vector<Eigen::Triplet<double>> trip;
      trip.reserve(fl.dq.size() + nu);
      for (size_t k = 0; k < nu; ++k) {
        const size_t e = static_cast<size_t>(impl_->elements[k]);
        const Element& el = m_.elements[e];
        const double c = m_.media[static_cast<size_t>(el.medium)].c(T[e], 0.0);
        R[static_cast<Eigen::Index>(k)] = (d.a0 * enthalpy(e, T[e]) + d.history[e]) / h - fl.q[e];
        trip.emplace_back(static_cast<int>(k), static_cast<int>(k), d.a0 * el.mass * c / h);
      }
      for (const Triplet& t : fl.dq) {
        const int r = impl_->unknown[static_cast<size_t>(t.row)], c = impl_->unknown[static_cast<size_t>(t.col)];
        if (r >= 0 && c >= 0) trip.emplace_back(r, c, -t.value);
      }
      Eigen::SparseMatrix<double> J(static_cast<Eigen::Index>(nu), static_cast<Eigen::Index>(nu));
      J.setFromTriplets(trip.begin(), trip.end());
      J.makeCompressed();
      lu.compute(J);
      if (lu.info() != Eigen::Success) return false;
      Eigen::VectorXd delta = lu.solve(-R);
      if (lu.info() != Eigen::Success || !delta.allFinite()) return false;
      // Keep temperatures positive: never move more than 90% towards zero.
      double scale = 1.0;
      for (size_t k = 0; k < nu; ++k) {
        const double Tk = T[static_cast<size_t>(impl_->elements[k])], dk = delta[static_cast<Eigen::Index>(k)];
        if (Tk + dk < 0.1 * Tk) scale = std::min(scale, 0.9 * Tk / -dk);
      }
      double max_step = 0;
      for (size_t k = 0; k < nu; ++k) {
        const double dk = scale * delta[static_cast<Eigen::Index>(k)];
        T[static_cast<size_t>(impl_->elements[k])] += dk;
        max_step = std::max(max_step, std::abs(dk));
      }
      for (double x : T) Tmax = std::max(Tmax, x);
      if (scale == 1.0 && max_step <= s_.tolerance * Tmax) {
        last = compute_flows(m_, impl_->emitters, T, d, false);
        return true;
      }
    }
    return false;
  };

  std::vector<double> T_new;
  Flows last;
  int iterations = 0;
  const bool bdf2 = s_.second_order && dt_prev_ > 0 && std::abs(dt - dt_prev_) <= 1e-12 * dt_prev_;
  bool ok = solve(T_, bdf2 ? &T_prev_ : nullptr, t_, dt, T_new, iterations, last);
  report.newton_iterations = iterations;
  if (!ok) {
    // Halve the step (backward Euler) until Newton converges; the radiation sample is reused.
    std::function<bool(const std::vector<double>&, double, double, int, std::vector<double>&)> cut =
        [&](const std::vector<double>& T0, double ta, double h, int depth, std::vector<double>& out) -> bool {
      int its = 0;
      if (solve(T0, nullptr, ta, h, out, its, last)) {
        report.newton_iterations += its;
        return true;
      }
      if (depth >= s_.max_cuts) return false;
      report.cuts = std::max(report.cuts, depth + 1);
      std::vector<double> mid;
      return cut(T0, ta, h / 2, depth + 1, mid) && cut(mid, ta + h / 2, h / 2, depth + 1, out);
    };
    std::vector<double> mid;
    report.cuts = 1;
    ok = cut(T_, t_, dt / 2, 1, mid) && cut(mid, t_ + dt / 2, dt / 2, 1, T_new);
    if (!ok)
      throw std::runtime_error("Newton's method did not converge at t = " + std::to_string(t_) + " s even after " +
                               std::to_string(s_.max_cuts) + " halvings of the step");
  }

  // Bookkeeping.
  PowerBalance& b = report.balance;
  b.emitted = last.emitted;
  b.absorbed = last.absorbed;
  if (rad_ptr)
    for (double x : rad.external) b.external += x;
  for (const HeatSource& h : m_.heat_sources) {
    const double per = h.power(0, t1) / static_cast<double>(h.elements.size());
    b.heat_sources += per * static_cast<double>(h.elements.size());
  }
  for (size_t e = 0; e < n; ++e) {
    if (impl_->unknown[e] < 0) b.into_fixed += last.q[e];
    else b.stored_rate += (enthalpy(e, T_new[e]) - enthalpy(e, T_[e])) / dt;
  }
  q_ = last.q;

  if (report.cuts == 0) {
    T_prev_ = T_;
    dt_prev_ = dt;
  } else {
    dt_prev_ = 0;  // history at another step size: restart with backward Euler
  }
  T_ = std::move(T_new);
  t_ = t1;
  ++step_;
  report.t = t_;
  return report;
}

}  // namespace th
