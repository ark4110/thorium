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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "tle_orbit.h"

#include <stdlib.h>

namespace ballistics  
{
  void tle_orbit_free(tle_orbit_t* o)
  {
    if (!o)
      return;
    free((void*)o->tle.header);
    free((void*)o->tle.line_1);
    free((void*)o->tle.line_2);
    free(o);
  }
}
