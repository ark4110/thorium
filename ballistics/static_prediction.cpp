// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// thorium is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// thorium is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with thorium.  If not, see <http://www.gnu.org/licenses/>.

#include "static_prediction.h"

#include <stdlib.h>

namespace ballistics_static
{
  using ballistics::clock_t;
  using ballistics::prediction_t;

  struct system_t : ballistics::system_t
  {
    params_t params;
  };

  int init(system_t* system, const params_t* params)
  {
    system->params = *params;

    return BALLISTICS_OK;
  }

  int shutdown(system_t* system)
  {
    return BALLISTICS_OK;
  }

  int predict_orbit(system_t* system, prediction_t* prediction)
  {
    *prediction = system->params.predictor_func(system->params.func_param);

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
