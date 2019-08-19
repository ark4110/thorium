// Copyright (c) 2015-2019 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

namespace cuda_math
{
  __global__ void reset(int n, int* data)
  {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for (; id < n; id += blockDim.x * gridDim.x)
    {
      data[id] = 0;
    }
  }
}
