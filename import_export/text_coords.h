// Copyright (c) 2022 Contributors as noted in the AUTHORS file.
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

#pragma once

#include "subject/system.h"

namespace texture_coordinates
{
  struct params_t
  {
    int only_emission = 1;
  };

  void generate(const subject::scene_t* scene, const float* temperatures, float* coordinates);
  int dump(const char* filename, int n_vertices, const float* coordinates);
  int dump_scene_with_colors(const char* filename, const char* colormap, const subject::scene_t* scene, const float* temperatures, params_t params);
}
