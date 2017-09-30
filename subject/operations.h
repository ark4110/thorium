// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "system.h"

namespace subject
{
  void scene_mesh_compose(scene_t* scene, int n_first_idx, int n_faces);
  void scene_mesh_translate(scene_t* scene, int n_mesh, math::vec3 translation);
  void scene_mesh_rotate(scene_t* scene, int n_mesh, math::vec3 from, math::vec3 to);

  // @note Completely brute-force algorithm for tests
  bool scenes_mesh_faces_near_enough(const scene_t* a, const scene_t* b, int ia, int ib);
}