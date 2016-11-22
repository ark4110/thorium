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
  @brief Emission state for Earth satellite.
  @detail Consider satellite orbit and attitude.
  @detail Supports Sun radiance and Earth shadow.
  @detail Supports Earth radiance as source of parallel rays.
  @todo Support Sun rays reflection from Earth.
*/

#pragma once


#include "system.h"
#include "../ballistics/system.h"

namespace planet_emission
{
  struct params_t : emission::params_t
  {
    float planet_radius; // [m]
    float planet_albedo; // [na]
    float planet_temperature; // [K]
    float solar_flux; // [W/m^2]

    ballistics::predictor_t predictor;

    int n_subdisivion;
    int n_rays;

    int parallel_rays_emitter_type;
  };
   
  emission::system_t* system_create();
}
