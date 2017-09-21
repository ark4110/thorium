// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
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

    int convert_block_size;

    subject::scene_t* scene;

    int n_hits;
    optix_hit_t* hits;
  };

  state_t* state_init(subject::scene_t* scene, bool force_cpu);
  void state_free(state_t* state);

  int state_prepare(state_t* state, ray_caster::task_t* task);

  int convert(state_t* state, ray_caster::task_t* task);
  
}
