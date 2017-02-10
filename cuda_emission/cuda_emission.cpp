// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

#include "cuda_emission.h"

#include <cuda_runtime_api.h>
#include "../cuda_ray_caster/task_relocation.h"

namespace cuda_emission
{
  bool is_supported()
  {
    int count = 0;
    return cudaSuccess == cudaGetDeviceCount(&count) && count > 0;
  }

  int cuda_task_relocation(emission::system_t*, emission::task_t* task, int device)
  {
    if (device == 0)
      device = 1; // @todo Use device id

    if (!task || !task->rays)
      return EMISSION_OK;

    if (task->rays->relocate)
      return task->rays->relocate(task->rays, device);

    // @todo How to avoid dependency with cuda_ray_caster?
    return cuda_task_relocation::relocate_task(task->rays, device);
  }
}
