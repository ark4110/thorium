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

#include "system.h"
#include "libpredict_tle.h"

#include <stdlib.h>

namespace ballistics
{
  system_t* system_create(int type, const params_t* params)
  {
    system_t* system = 0;
    switch (type)
    {
    case BALLISTICS_TLE:
      system = libpredict_tle::system_create();
      break;

    default:
      return 0;
    }

    system_init(system, params);

    return system;
  }

  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  int system_init(system_t* system, const params_t* params)
  {
    return system->methods->init(system, params);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int predict_orbit(system_t* system, const clock_t* clock, prediction_t* prediction)
  {
    return system->methods->predict_orbit(system, clock, prediction);
  }
}