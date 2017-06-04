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
#include <string.h>
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

    char line1[1024];
    char line2[1024];

    strncpy(line1, params->orbit.tle.line_1, sizeof(line1));
    strncpy(line2, params->orbit.tle.line_2, sizeof(line2));

    char *tle[2] = { line1, line2 };
    system->elements = predict_parse_tle(tle);

    return BALLISTICS_OK;
  }

  int shutdown(system_t* system)
  {
    if (system->elements)
      predict_destroy_orbital_elements(system->elements);

    return BALLISTICS_OK;
  }

  int predict_orbit(system_t* system, prediction_t* prediction)
  {
    time_t time = system->params.clock.func(system->params.clock.param);
    predict_julian_date_t julian_time = predict_to_julian(time);

    struct predict_orbit orbit;
    ::predict_orbit(system->elements, &orbit, julian_time);

    double eci_sun_position[3];
    sun_predict(julian_time, eci_sun_position);

    math::vec3 eci_satellite_position = math::make_vec3((float)orbit.position[0], (float)orbit.position[1], (float)orbit.position[2]) * 1000;
    math::vec3 sun_position = math::make_vec3((float)eci_sun_position[0], (float)eci_sun_position[1], (float)eci_sun_position[2]) * 1000 - eci_satellite_position;

    math::vec3 velocity_direction = math::normalize(math::make_vec3((float)orbit.velocity[0], (float)orbit.velocity[1], (float)orbit.velocity[2]));
    math::vec3 earth_direction = math::normalize(-eci_satellite_position);

    math::vec3 eci_direction;
    switch (system->params.orbit.attitude.type)
    {
    case ATTITUDE_TYPE_NADIR:
      eci_direction = earth_direction;
    break;

    case ATTITUDE_TYPE_SUN:
      eci_direction = math::normalize(sun_position);
    break;

    case ATTITUDE_TYPE_ORBIT:
      eci_direction = velocity_direction;
    break;

    default:
      return -BALLISTICS_ERROR;
    }

    // @todo Support custom basis for attitude
    math::vec3 attitude_axis = math::make_vec3(0, 0, 1);

    math::mat33 transformation = math::rotate_towards(eci_direction, attitude_axis);
    switch (system->params.orbit.attitude.type)
    {
    case ATTITUDE_TYPE_NADIR:
    case ATTITUDE_TYPE_ORBIT:
    {
      math::vec3 orbit_plane_direction = cross(velocity_direction, earth_direction);
      math::vec3 transformed_orbit_normal = transformation * orbit_plane_direction;
      math::mat33 axis2_transformation = math::rotate_towards(transformed_orbit_normal, math::make_vec3(0, 1, 0));
      transformation = axis2_transformation * transformation;
    }
      break;

    case ATTITUDE_TYPE_SUN:
      break;

    default:
      return -BALLISTICS_ERROR;
    }
 
    prediction->earth_position = transformation * (-eci_satellite_position);
    prediction->sun_position = transformation * sun_position;
    prediction->eclispsed = orbit.eclipsed;

    return BALLISTICS_OK;
  }

  const ballistics::system_methods_t methods =
  {
    (int(*)(ballistics::system_t* system, const ballistics::params_t*))&init,
    (int(*)(ballistics::system_t* system))&shutdown,
    (int(*)(ballistics::system_t* system, prediction_t* prediction))&predict_orbit
  };

  ballistics::system_t* system_create()
  {
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
