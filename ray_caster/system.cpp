// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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

/**
 * This module contains basic types to represent a scene for ray caster calculation.
 * Module also contains base type (system_t) for ray caster with table of virtual methods.
 */

#include "system.h"
#include "naive_cpu.h"
#include "../cuda_ray_caster/cuda_system.h"
#include "../cuda_ray_caster/optix_system.h"
#include "../math/ray.h"
#include <stdlib.h>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
	

namespace ray_caster
{
  void host_rays_data_free(rays_data_t* data)
  {
    free(data->ray);
    free(data->hit_face_idx);
    free(data->hit_distance);
    free(data->power);
  }

  void host_task_free_data(task_t* task)
  {
    host_rays_data_free(task);
    host_rays_data_free(&task->next_gen);
  }

  void host_rays_data_alloc(rays_data_t* data, int n_rays)
  {
    data->ray = (math::ray_t*) calloc(n_rays, sizeof(math::ray_t));
    data->hit_face_idx = (int*)calloc(n_rays, sizeof(int));
    data->hit_distance = (float*)calloc(n_rays, sizeof(float));
    data->power = (float*)calloc(n_rays, sizeof(float));
  }

  void host_rays_data_grow(rays_data_t* data, int n_reserved)
  {
    data->ray = (math::ray_t*)realloc(data->ray, n_reserved * sizeof(math::ray_t));
    data->hit_face_idx = (int*)realloc(data->hit_face_idx, n_reserved * sizeof(int));
    data->hit_distance = (float*)realloc(data->hit_distance, n_reserved * sizeof(float));
    data->power = (float*)realloc(data->power, n_reserved * sizeof(float));
  }

  task_t* task_create(int n_rays)
  {
    task_t* task = (task_t*) calloc(1, sizeof(task_t));
    task->n_tasks = n_rays;
    task->n_reserved = n_rays;
    task->allocation_device = -1;
    task->free_data = &host_task_free_data;

    host_rays_data_alloc(task, n_rays);
    host_rays_data_alloc(&task->next_gen, n_rays);
    
    return task;
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

    if (valid_task->grow)
      return valid_task->grow(valid_task, n_rays);

    if (valid_task->allocation_device != -1)
      throw std::invalid_argument("Can't grow invalid task: external grow method not defined for device task.");

    int n_prev = valid_task->n_tasks;
    valid_task->n_tasks += n_rays;

    if (valid_task->n_reserved < valid_task->n_tasks)
    {
      int n_from = valid_task->n_reserved;
      valid_task->n_reserved = std::min<int>(valid_task->n_tasks * 2, valid_task->n_tasks + 100*1024);
      if (valid_task->n_reserved > 100000)
        fprintf(stderr, "Task grow from %d to %d rays\n", n_from, valid_task->n_reserved);
      
      host_rays_data_grow(valid_task, valid_task->n_reserved);
      host_rays_data_grow(&valid_task->next_gen, valid_task->n_reserved);
    }

    return n_prev;
  }

  int task_reserve(task_t** task, int n_rays)
  {
    int n_prev = task_grow(task, n_rays);
    (*task)->n_tasks = n_prev;
    
    return (*task)->n_reserved - (*task)->n_tasks;
  }

  int task_get_reserve(task_t* task)
  {
    return task ? (task->n_reserved - task->n_tasks) : 0;
  }

  int task_get_n_tasks(task_t* task)
  {
    return task ? task->n_tasks : 0;
  }

  void task_next_gen(task_t* task)
  {
    std::swap(task->next_gen, *(rays_data_t*)task);
    task->next_gen.n_tasks = 0;
  }

  void task_free_data(task_t* task)
  {
    if (task && task->free_data)
      task->free_data(task);
  }

  void task_free(task_t* task)
  {
    if (!task)
      return;

    if (task->n_reserved < 0) // check for static task
      return;

    if (task->free_data)
      task->free_data(task);
    free(task);
  }

  int cpu_fallback(int type)
  {
    if (cuda_ray_caster::is_supported())
      return type;

    switch (type)
    {
    case RAY_CASTER_NAIVE_CUDA:
      fprintf(stderr, "No CUDA detected. Fallback ray caster to CPU.\n");
    case RAY_CASTER_NAIVE_CPU:
    
      return RAY_CASTER_NAIVE_CPU;      
      break;

    case RAY_CASTER_OPTIX:
#if !defined _WIN32 || defined _WIN64
      return RAY_CASTER_OPTIX;
#else
    {
      fprintf(stderr, "No Optix engine in 32-bit. Fallback to NAIVE CUDA.\n");
      return cpu_fallback(RAY_CASTER_NAIVE_CUDA);
    }
    break;
#endif    

    default:
      return type;
    }
  }

  system_t* system_create(int type)
  {
    type = cpu_fallback(type);

    system_t* system = 0;
    switch (type)
    {
    case RAY_CASTER_NAIVE_CPU:
      system = raycaster_naive_cpu::system_create();
      break;

    case RAY_CASTER_NAIVE_CUDA:
      system = naive_cuda_ray_caster::system_create();
      break;

#if !defined _WIN32 || defined _WIN64
    case RAY_CASTER_OPTIX:
      system = optix_ray_caster::system_create(false);
      break;
    
    case RAY_CASTER_OPTIX_CPU:
      system = optix_ray_caster::system_create(true);
      break;
#endif

    default:
      fprintf(stderr, "Unknown ray caster system type %d.\n", type);
    }

    system_init(system);
    
    return system;
  }

  const char* system_names[RAY_CASTER_SYSTEMS_COUNT + 1] = {
    "naive cpu",
    "naive cuda",
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

  int system_set_scene(system_t* system, subject::scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_update(system_t* system)
  {
    return system->methods->update(system);
  }

  int system_cast(system_t* system, task_t* task)
  {
    return system->methods->cast(system, task);
  }

  int system_relocate_task(system_t* system, task_t* task, int device)
  {
    // @todo Add dummy relocation for depricated systems?
    if (system->methods->relocate_task)
      return system->methods->relocate_task(system, task, device);
    else
      return RAY_CASTER_OK;
  }

  math::vec3 task_hit_point(const task_t* task, int n_ray)
  {
    return math::ray_point(task->ray[n_ray], task->hit_distance[n_ray]);
  }
}
