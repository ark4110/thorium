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
 * This module contains basic types to represent a scene for ray caster calculation.
 * Module also contains base type (system_t) for ray caster with table of virtual methods.
 */

#include "system.h"
#include "naive_cpu.h"
#include "zgrid_cpu.h"
#include "../cuda_ray_caster/cuda_system.h"
#include "../optix_ray_caster/optix_system.h"
#include <stdlib.h>
#include <stdexcept>
	

namespace ray_caster
{
  scene_t* scene_create()
  {
    scene_t* s = (scene_t*)malloc(sizeof(scene_t));
    *s = { 0, 0 };
    return s;
  }

  void scene_free(scene_t* scene)
  {
    if (scene)
      free(scene->faces);
    free(scene);
  }

  task_t* task_create(int n_rays)
  {
    task_t* task = (task_t*) malloc(sizeof(task_t));
    task->n_tasks = n_rays;
    task->n_reserved = n_rays;
    task->ray = (math::ray_t*)     malloc(n_rays * sizeof(math::ray_t));
    task->hit_face = (face_t**)    malloc(n_rays * sizeof(ray_caster::face_t*));
    task->hit_point = (math::vec3*)malloc(n_rays * sizeof(math::vec3));
    return task;
  }

  task_t task_static(int n_rays, ray_t* ray, face_t** hit_face, math::vec3* hit_point)
  {
    return{ n_rays, -1, ray, hit_face, hit_point };
  }

  int task_grow(task_t** task, int n_rays)
  {
    if (!*task)
    {
      *task = task_create(n_rays);
      return 0;
    }

    task_t* valid_task = *task;

    if (valid_task->n_reserved < 0)
      throw std::invalid_argument("Can't grow static task.");

    int n_prev = valid_task->n_tasks;
    valid_task->n_tasks += n_rays;

    if (valid_task->n_reserved < valid_task->n_tasks)
    {
      valid_task->n_reserved = valid_task->n_tasks;
      valid_task->ray = (math::ray_t*)realloc(valid_task->ray, valid_task->n_tasks * sizeof(math::ray_t));
      valid_task->hit_face = (face_t**)realloc(valid_task->hit_face, valid_task->n_tasks * sizeof(ray_caster::face_t*));
      valid_task->hit_point = (math::vec3*)realloc(valid_task->hit_point, valid_task->n_tasks * sizeof(math::vec3));
    }

    return n_prev;
  }

  int task_reserve(task_t** task, int n_rays)
  {
    if (!*task)
    {
      *task = task_create(n_rays);
      (*task)->n_tasks = 0;
      return n_rays;
    }

    task_t* valid_task = *task;

    if (valid_task->n_reserved < 0)
      throw std::invalid_argument("Can't grow static task.");

    int n_prev = valid_task->n_reserved - valid_task->n_tasks;
    if (n_prev >= n_rays)
      return n_prev;

    valid_task->n_reserved = valid_task->n_tasks + n_rays;
    valid_task->ray = (math::ray_t*)realloc(valid_task->ray, valid_task->n_reserved * sizeof(math::ray_t));
    valid_task->hit_face = (face_t**)realloc(valid_task->hit_face, valid_task->n_reserved * sizeof(ray_caster::face_t*));
    valid_task->hit_point = (math::vec3*)realloc(valid_task->hit_point, valid_task->n_reserved * sizeof(math::vec3));
    return n_rays;
  }

  int task_get_reserve(task_t* task)
  {
    return task ? (task->n_reserved - task->n_tasks) : 0;
  }

  int task_get_n_tasks(task_t* task)
  {
    return task ? task->n_tasks : 0;
  }

  int task_reset(task_t* task)
  {
    if (task == 0)
      return 0;
    
    int prev = task->n_tasks;
    task->n_tasks = 0;
    return prev;
  }

  void task_free(task_t* task)
  {
    if (task && task->n_reserved >= 0)
    {
      free(task->ray);
      free(task->hit_face);
      free(task->hit_point);
      free(task);
    }
  }

  system_t* system_create(int type)
  {
    system_t* system = 0;
    switch (type)
    {
    case RAY_CASTER_NAIVE_CPU:
      system = raycaster_naive_cpu::system_create();
      break;

    case RAY_CASTER_NAIVE_CUDA:
      system = naive_cuda_ray_caster::system_create();
      break;

    case RAY_CASTER_ZGRID_CPU:
      system = ray_caster_zgrid_cpu::system_create();
      break;

    case RAY_CASTER_ZGRID_CUDA:
      system = zgrid_cuda_ray_caster::system_create();
      break;

#if !(defined _WIN32 && !defined _WIN64)
    case RAY_CASTER_OPTIX:
      system = optix_ray_caster::system_create();
      break;    
#endif
    }    

    system_init(system);
    
    return system;
  }

  const char* system_names[RAY_CASTER_SYSTEMS_COUNT + 1] = {
    "naive cpu",
    "naive cuda",
    "zgrid cpu",
    "zgrid cuda",
    "optix",
    "unknown"
  };

  const char* system_get_name(int type)
  {
    if (type >= 0 && type < RAY_CASTER_SYSTEMS_COUNT)
      return system_names[type];
    else
      return system_names[RAY_CASTER_SYSTEMS_COUNT];
  }

  const char* system_ids[RAY_CASTER_SYSTEMS_COUNT + 1] = {
    "naive_cpu",
    "naive_cuda",
    "zgrid_cpu",
    "zgrid_cuda",
    "optix",
    "unknown"
  };

  const char* system_get_id(int type)
  {
    if (type >= 0 && type < RAY_CASTER_SYSTEMS_COUNT)
      return system_ids[type];
    else
      return system_ids[RAY_CASTER_SYSTEMS_COUNT];
  }

  system_t* system_create_default()
  {
    return system_create(RAY_CASTER_NAIVE_CPU);
  }

  void system_free(system_t* system)
  { 
    system_shutdown(system);
    free(system);
  }

  int system_init(system_t* system)
  {
    return system->methods->init(system);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_prepare(system_t* system)
  {
    return system->methods->prepare(system);
  }

  int system_cast(system_t* system, task_t* task)
  {
    return system->methods->cast(system, task);
  }
}
