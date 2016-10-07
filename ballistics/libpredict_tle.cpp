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

#include "libpredict_tle.h"
#include "../math/mat.h"

#include <stdlib.h>
#include <predict/predict.h>

namespace libpredict_tle
{
  using ballistics::clock_t;
  using ballistics::prediction_t;

  struct system_t : ballistics::system_t
  {
    params_t params;
    predict_orbital_elements_t* elements;
  };

  int init(system_t* system, const params_t* params)
  {
    system->params = *params;

    char *tle[2] = { const_cast<char*>(params->tle_line_1), const_cast<char*>(params->tle_line_2) };
    system->elements = predict_parse_tle(tle);

    return BALLISTICS_OK;
  }

  int shutdown(system_t* system)
  {
    if (system->elements)
      predict_destroy_orbital_elements(system->elements);

    return BALLISTICS_OK;
  }

  int predict_orbit(system_t* system, const clock_t* clock, prediction_t* prediction)
  {
    predict_julian_date_t julian_time = predict_to_julian(clock->utc);

    struct predict_orbit orbit;
    ::predict_orbit(system->elements, &orbit, julian_time);

    double eci_sun_position[3];
    sun_predict(julian_time, eci_sun_position);

    math::vec3 eci_satellite_position = math::make_vec3(orbit.position[0], orbit.position[1], orbit.position[2]) * 1000;
    math::vec3 sun_position = math::make_vec3(eci_sun_position[0], eci_sun_position[1], eci_sun_position[2]) * 1000 - eci_satellite_position;

    math::vec3 eci_direction;
    switch (system->params.orientation.type)
    {
    case ORIENTATION_TYPE_NADIR:
    {
      math::vec3 earth_direction = math::normalize(-eci_satellite_position);
      eci_direction = earth_direction;
    }
    break;

    case ORIENTATION_TYPE_SUN:
    {
      math::vec3 sun_direction = math::normalize(sun_position);
      eci_direction = sun_direction;
    }
    break;

    case ORIENTATION_TYPE_ORBIT:
    {
      math::vec3 velocity_direction = math::normalize(math::make_vec3(orbit.velocity[0], orbit.velocity[1], orbit.velocity[2]));
      eci_direction = velocity_direction;
    }
    break;

    default:
      return -BALLISTICS_ERROR;
    }

    // @todo Support custom basis for orientation
    math::vec3 orientation_axis = math::make_vec3(0, 0, 1);

    math::mat33 orientation_transformation = math::rotate_towards(eci_direction, orientation_axis);

    prediction->earth_position = orientation_transformation * (-eci_satellite_position);
    prediction->sun_position = orientation_transformation * sun_position;
    prediction->eclispse_depth = orbit.eclipsed ? orbit.eclipse_depth : 0;

    math::vec3 sun_back = (math::rotate_towards(orientation_axis, eci_direction) * prediction->sun_position + eci_satellite_position) / 1000;

    sun_back.x -= eci_sun_position[0];
    sun_back.y -= eci_sun_position[1];
    sun_back.z -= eci_sun_position[2];

    math::vec3 sat_back = (math::rotate_towards(orientation_axis, eci_direction) * prediction->earth_position) / -1000;
    sat_back.x -= orbit.position[0];
    sat_back.y -= orbit.position[1];
    sat_back.z -= orbit.position[2];

    return BALLISTICS_OK;
  }

  const ballistics::system_methods_t methods =
  {
    (int(*)(ballistics::system_t* system, const ballistics::params_t*))&init,
    (int(*)(ballistics::system_t* system))&shutdown,
    (int(*)(ballistics::system_t* system, const clock_t* clock, prediction_t* prediction))&predict_orbit
  };

  ballistics::system_t* system_create()
  {
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
