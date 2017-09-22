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
 * This module contains CPU single-threaded implementation of ray caster.
 */

#include "naive_cpu.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include <limits>
#include <stdlib.h>

namespace raycaster_naive_cpu
{
  struct cpu_system_t : ray_caster::system_t
  {
    subject::scene_t* scene;
  };

  int init(cpu_system_t* system)
  {
    system->scene = 0;
    return RAY_CASTER_OK;
  }

  int shutdown(cpu_system_t* system)
  {
    return RAY_CASTER_OK;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;
    return RAY_CASTER_OK;
  }

  int update(cpu_system_t* system)
  {
    if (system->scene == 0 || system->scene->n_faces == 0)
      return -RAY_CASTER_ERROR;
    return RAY_CASTER_OK;
  }

  void task_drop_missed(ray_caster::task_t* task)
  {
    if (!task->n_tasks)
      return;

    int n_missed = 0;
    int end_idx = task->n_tasks - 1;
    while (end_idx >= 0 && task->hit_distance[end_idx] < 0)
    {
      ++n_missed;
      --end_idx;
    }

    if (end_idx < 0)
    {
      task->n_missed = n_missed;
      task->n_tasks = 0;
      return;
    }
    
    int i = 0;
    for (; i < end_idx; ++i)
    {
      if (task->hit_distance[i] < 0)
      {
        ++n_missed;
        task->ray[i] = task->ray[end_idx];
        task->hit_face_idx[i] = task->hit_face_idx[end_idx];
        task->hit_distance[i] = task->hit_distance[end_idx];
        task->power[i] = task->power[end_idx];
        --end_idx;
        while (end_idx != i && task->hit_distance[end_idx] < 0)
        {
          ++n_missed;
          --end_idx;
        }
      }
    }

    task->n_tasks = i + 1;
    task->n_missed = n_missed;
  }

  int cast(cpu_system_t* system, ray_caster::task_t* task)
  {
    using namespace ray_caster;
    for (int t = 0; t != task->n_tasks; ++t)
    {
      // for every ray
      math::ray_t ray = task->ray[t];
      math::point_t min_distance = std::numeric_limits<math::point_t>::max();
      task->hit_face_idx[t] = -1;
      for (int f = 0; f != system->scene->n_faces; ++f)
      {
        // and any faces from scene
        math::triangle_t triangle = system->scene->faces[f];
        float distance = triangle_intersect(ray, triangle);
        if (distance > 0 && distance < min_distance)
        {
          min_distance = distance;
          task->hit_face_idx[t] = (int)(&system->scene->faces[f] - system->scene->faces);
          task->hit_distance[t] = distance;
        }
      }
    }

    task_drop_missed(task);

    return RAY_CASTER_OK;
  }

  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&update,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  ray_caster::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
