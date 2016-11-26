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

#include "attitude.h"

namespace ballistics
{
  struct tle_t
  {
    char* header;
    char* line_1;
    char* line_2;
  };

  struct tle_orbit_t
  {
    attitude_t attitude;
    tle_t tle;
  };

  void tle_orbit_free(tle_orbit_t* o);
}
