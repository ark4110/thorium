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

#include "attitude.h"

namespace ballistics
{
  const char unknown_name[] = "unknown";
  const char nadir_name[] = "nadir";
  const char sun_name[] = "sun";
  const char orbit_name[] = "orbit";

  const char* attitude_get_name(int type)
  {
    switch (type)
    {
    case ATTITUDE_TYPE_NADIR:
      return nadir_name;
      break;

    case ATTITUDE_TYPE_SUN:
      return sun_name;
      break;

    case ATTITUDE_TYPE_ORBIT:
      return orbit_name;
      break;

    default:
      return unknown_name;
    }
  }
}
