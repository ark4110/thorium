// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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

/**
 * This module contains basic types to represent a scene for form factors calculation.
 * Module also contains base type (system_t) for form factor calculation with table of virtual methods.
 */

#include "malley_cpu.h"
#include "parallel_rays_cpu.h"
#include "system.h"
#include <cstdlib>

namespace emission
{
  task_t* task_create()
  {
    task_t* task = (task_t*)calloc(1, sizeof(task_t));
    return task;
  }

  void task_free(task_t* task)
  {
    if (task)
    {
      task_free(task->rays);
      free(task->power);
    }

    free(task);
  }

  int task_grow(task_t* task, int n_rays)
  {
    int n_input_rays = task->rays ? task->rays->n_tasks : 0;    
    task->rays = ray_caster::task_realloc(task->rays, n_input_rays + n_rays);
    task->power = (float*)realloc(task->power, (n_input_rays + n_rays) * sizeof(float));
    return n_input_rays;
  }

  system_t* system_create(int type, const params_t* params)
  {
    system_t* system = 0;
    switch (type)
    {
    case EMISSION_MALLEY_CPU:
      system = malley_cpu::system_create();
      break;

    case EMISSION_PARALLEL_RAYS_CPU:
      system = parallel_rays_emission_cpu::system_create();
      break;

    default:
      return 0;
    }

    system_init(system, params);

    return system;
  }

  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  int system_init(system_t* system, const params_t* params)
  {
    return system->methods->init(system, params);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, subject::scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_calculate(system_t* system, task_t* task)
  {
    return system->methods->calculate(system, task);
  }
}
