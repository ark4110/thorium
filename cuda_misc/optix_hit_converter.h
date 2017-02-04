// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
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
  Converts Optix ray hit info into world coordinates.
*/
#pragma once

#include "../ray_caster/system.h"

namespace optix_hit_converter
{
  bool is_supported();

  struct optix_hit_t
  {
    float distance;
    int triangle_id;
    float u;
    float v;
  };

  struct state_t
  {
    int optix_buffer_type;

    ray_caster::face_t* host_faces;
    void* device_faces;

    int n_hits;
    optix_hit_t* hits;

    void* device_hit_points;
    void** device_hit_faces;
  };

  state_t* state_init(const ray_caster::scene_t* scene);
  void state_free(state_t* state);

  void state_prepare(state_t* state, ray_caster::task_t* task);

  int convert(state_t* state, ray_caster::task_t* task);
  
}
