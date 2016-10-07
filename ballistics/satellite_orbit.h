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


#pragma once

namespace satellite_orbit
{
  enum orientation_axis_t
  {
    orientation_axis_x,
    orientation_axis_y,
    orientation_axis_z,
  };

  enum orientation_type_t
  {
    nadir,
    sun,
    orbit
  };

  struct description_t
  {
    char* tle_title;
    char* tle_line_1;
    char* tle_line_2;
    int orientation_axis;
    int orientation_type;
  };
}
