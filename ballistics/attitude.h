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

#pragma once

namespace ballistics
{
#define ATTITUDE_TYPE_NADIR 0
#define ATTITUDE_TYPE_SUN 1
#define ATTITUDE_TYPE_ORBIT 2

  /**
    @brief Satellite attitude on orbit.
    @detail Due to satellite attitude preferences it will rotate on orbit.
    @detail Attitude axis is scene Z-axis.
  */
  struct attitude_t
  {
    int type;
  };

  const char* attitude_get_name(int type);
}
