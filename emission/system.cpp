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
#include "planet_emission.h"
#include "system.h"
#include "../cuda_emission/cuda_emission.h"
#include <cstdlib>
#include <cstdio>

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
    int n_prev_reserved = task->rays ? task->rays->n_reserved : 0;
    int n_input_rays = ray_caster::task_grow(&task->rays, n_rays);
    int n_total = task->rays->n_tasks;
    if (n_prev_reserved != task->rays->n_reserved)
      task->power = (float*)realloc(task->power, task->rays->n_reserved * sizeof(float));
    return n_input_rays;
  }

  int task_reserve(task_t* task, int n_rays)
  {
    int n_prev_reserved = task->rays ? task->rays->n_reserved : 0;
    int n_current_reserve = ray_caster::task_reserve(&task->rays, n_rays);
    int n_target_reserve = task->rays->n_tasks + n_rays;
    if (n_prev_reserved < n_target_reserve)
      task->power = (float*)realloc(task->power, n_target_reserve * sizeof(float));
    return n_current_reserve;
  }

  void task_reset(task_t* task)
  {
    task_reset(task->rays);
  }

  void task_add(task_t* task, const ray_caster::ray_t& ray, float power)
  {
    if (task_get_reserve(task->rays) == 0)
      task_reserve(task, 1024*128);

    int& n_rays = task->rays->n_tasks;
    task->rays->ray[n_rays] = ray;
    task->power[n_rays] = power;
    ++n_rays;
  }

  int cpu_fallback(int type)
  {
    if (cuda_emission::is_supported())
      return type;

    fprintf(stderr, "No CUDA detected. Fallback emission to CPU.\n");

    switch (type)
    {
    case EMISSION_MALLEY_CPU:
    case EMISSION_MALLEY_CUDA:
      return EMISSION_MALLEY_CPU;
      break;

    case EMISSION_PARALLEL_RAYS_CPU:
    case EMISSION_PARALLEL_RAYS_CUDA:
      return EMISSION_PARALLEL_RAYS_CPU;
      break;

    case EMISSION_PLANET_CPU:
      return EMISSION_PLANET_CPU;
      break;

    default:
      return -1;
    }
  }

  system_t* system_create(int type, const params_t* params)
  {
    type = cpu_fallback(type);

    system_t* system = 0;
    switch (type)
    {
    case EMISSION_MALLEY_CPU:
      system = malley_cpu::system_create();
      break;

    case EMISSION_PARALLEL_RAYS_CPU:
      system = parallel_rays_emission_cpu::system_create();
      break;

    case EMISSION_MALLEY_CUDA:
      system = cuda_malley_emission::system_create();
      break;

    case EMISSION_PARALLEL_RAYS_CUDA:
      system = cuda_parallel_rays_emission::system_create();
      break;

    case EMISSION_PLANET_CPU:
      system = planet_emission::system_create();
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
