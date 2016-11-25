// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
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
  @brief Ballistics system define satellite trajectory.
  @detail Coordinate system for prediction is scene coordinate system.
  @detail I.e. satellite coordinate system is the center of the world, Earth and Sun are rotating around it.
*/

#pragma once

#include "attitude.h"
#include "../math/types.h"

#include <time.h>

namespace ballistics
{
  /// @brief Orbit prediction in satellite coordinate system.
  struct prediction_t
  {
    math::vec3 earth_position; // [m]
    math::vec3 sun_position; // [m]
    int eclispsed;
  };

  /// @brief Clock report current time in UTC.
  struct clock_t
  {
    time_t(*func)(void* param);
    void* param;
  };

  struct system_t
  {
    const struct system_methods_t* methods;
  };

  struct params_t
  {
    clock_t clock;
    attitude_t attitude;
  };

#define BALLISTICS_OK    0
#define BALLISTICS_ERROR 500

  struct system_methods_t
  {
    int(*init)(system_t* system, const params_t* params);
    int(*shutdown)(system_t* system);
    int(*predict_orbit)(system_t* system, prediction_t* prediction);
  };

/**
  NORAD two-line elements set ballistics based on libpredict.
  ECI (Earth-centered inertial) TEME (true equator, mean equinox) coordinates will be transformed using satellite attitude parameters.
*/
#define BALLISTICS_TLE 1

/// @note Static prediction params_t does not inherit base ballistics::params_t!
#define BALLISTICS_STATIC 2

  system_t* system_create(int type, const params_t* params);

  void system_free(system_t* system);
  int system_init(system_t* system, const params_t* params);
  int system_shutdown(system_t* system);
  int predict_orbit(system_t* system, prediction_t* prediction);
}