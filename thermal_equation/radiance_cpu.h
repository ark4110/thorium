// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

/**
 * This module contains thermal equation for Stefan-Boltzman radiation using Monte-Carlo method.
 * Calculate faces weights based on power emission.
 * Calculate power emission/absorption on each step.
 */

#pragma once

#include "../thermal_equation/system.h"
#include "../emission/system.h"

namespace radiance_equation
{
  struct stats_t
  {
    union
    {
      struct
      { 
        int n_absorbed;
        int n_reflected;
        int n_diffused;
        int n_transmited;
        int n_droped;
        int n_postponed;
        int n_missed;
      } named;
      int raw[7];
    } quants;

    union
    {
      struct
      {        
        float e_absorbed;
        float e_reflected;
        float e_diffused;
        float e_transmited;
        float e_droped;
        float e_postponed;
        float e_missed;
      } named;
      float raw[7];
    } powers;
  };

  struct params_t
  {
    int n_emitters;
    emission::system_t** emitters;

    ray_caster::system_t* ray_caster;

    // Optional stats collector.
    stats_t* stats;
  };

  thermal_equation::system_t* system_create();

  void print_stats(const stats_t& stats);
}
