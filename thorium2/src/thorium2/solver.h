// Thorium 2 — implicit time integration.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Energy balance of element i (free elements; T_SET elements follow their prescribed T(t)):
//
//   m_i dH_i(T_i)/dt = sum_junctions g (T_j - T_i)                       conduction
//                    + sum_f F(f->i) eps_f sigma A_f T_e(f)^4            absorbed radiation
//                    - sum_{f of i} eps_f sigma A_f T_i^4                emitted radiation
//                    + Q_ext,i(t) + Q_heat,i(t)                          sources
//
// H(T) = integral of c(T) dT is the specific enthalpy, so a temperature-dependent heat capacity
// conserves energy exactly (version 1 used d(cT)/dt, which is wrong when c depends on T).
// The time derivative is discretised by BDF2 (first step and steps after a cut: backward
// Euler); both are L-stable, so the step is limited by accuracy only. The nonlinear system is
// solved by Newton's method with a sparse LU factorisation; the fractions F(f->i) are sampled
// by Monte Carlo once per step at the temperatures of the beginning of the step.
#pragma once

#include "thorium2/model.h"
#include "thorium2/radiation.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace th {

struct SolverSettings {
  std::uint64_t seed = 1;
  int photons_per_face = 0;      ///< 0: from I_PARAM.
  int max_newton = 30;
  double tolerance = 1e-9;       ///< Newton stops when max |dT| < tolerance * max T.
  int max_cuts = 12;             ///< A step that does not converge is halved up to this many times.
  bool second_order = true;      ///< BDF2 (true) or backward Euler only (false).
};

/// Power terms of the whole model at the end of a step, W.
///
/// With backward Euler the terms close exactly:
///   stored_rate + into_fixed = external + heat_sources + absorbed - emitted.
/// With BDF2 stored_rate is the plain difference quotient over the step, so the identity holds
/// only up to the time discretisation error.
struct PowerBalance {
  double external = 0;           ///< Absorbed from RSOURCE1 by all elements.
  double heat_sources = 0;       ///< HSOURCE1, all elements.
  double emitted = 0;            ///< Emitted by all radiating faces (free and fixed elements).
  double absorbed = 0;           ///< Of the emitted power, absorbed by elements of the model.
  double into_fixed = 0;         ///< Net power into T_SET elements (what the "thermostat" removes).
  double stored_rate = 0;        ///< Change of the energy of free elements over the step / dt.
};

struct StepReport {
  double t = 0;                  ///< Time at the end of the step.
  int newton_iterations = 0;
  int cuts = 0;                  ///< How many times the step had to be halved.
  long long bounce_limit_hits = 0;
  PowerBalance balance;
};

class Solver {
public:
  explicit Solver(const Model& model, SolverSettings settings = {});
  ~Solver();
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  double time() const { return t_; }
  bool finished() const;
  const std::vector<double>& temperatures() const { return T_; }
  /// Net heat flow into each element at the end of the last step, W.
  const std::vector<double>& heat_flow() const { return q_; }

  /// Advances one step of I_PARAM t_STEP (the last one may be shorter to end at t_end).
  StepReport step();

private:
  struct Impl;
  const Model& m_;
  SolverSettings s_;
  double t_ = 0;
  std::vector<double> T_, T_prev_, q_;
  double dt_prev_ = 0;
  std::uint64_t step_ = 0;
  std::unique_ptr<RadiationScene> scene_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace th
