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
* This module contains emission generator for parallel rays.
*/

#pragma once

#include "../emission/system.h"

namespace parallel_rays_emission_cpu
{
  /**
    @brief Effective area of rays emission.
    @detail Parallel rays emitter calculates scene bounding sphere and scale it by this factor.
        It is required to ensure that all parts of the scene may be hit by emitted rays.
    @todo Reduce factor below 1.1 because it means that 10% of rays are emitted to empty space by definition.
  */
  const float scene_bsphere_emission_scale = 1.1f;

  struct parallel_rays_source_t
  {
    int n_rays;
    float power; /// W/m^2
    math::vec3 direction;
  };

  /// @todo Provide a way to change origin and direction in time.
  struct params_t : emission::params_t
  {
    int n_sources;
    parallel_rays_source_t* sources;
  };

  /**
  *  @brief Factory method to create parallel rays emission.
  *  @detail Emit randomly (uniformly) distributed parallel rays from specified plane in direction of plane normal.
  */
  emission::system_t* system_create();
}
