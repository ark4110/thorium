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


#include "radiance_cpu.h"
#include "../math/triangle.h"
#include <float.h>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace radiance_equation
{
  struct cpu_system_t : thermal_equation::system_t
  {    
    params_t params;

    subject::scene_t* scene;

    emission::task_t* emission_task;

    int* face_to_mesh_index;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, params_t* params)
  {    
    system->params = *params;

    system->scene = 0;
    system->emission_task = 0;
    system->face_to_mesh_index = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    emission::task_free(system->emission_task);
    system->emission_task = 0;

    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;

    return THERMAL_EQUATION_OK;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;
    ray_caster::scene_t* ray_caster_scene = (ray_caster::scene_t*)scene; // slicing binary compatible scenes

    int r = 0;
    if ((r = system_set_scene(system->params.ray_caster, ray_caster_scene)) < 0)
      return r;

    if ((r = system_prepare(system->params.ray_caster)) < 0)
      return r;

    for (int e = 0; e != system->params.n_emitters; ++e)
    if ((r = system_set_scene(system->params.emitters[e], scene)) < 0)
      return r;

    emission::task_free(system->emission_task);
    system->emission_task = emission::task_create();

    // @todo Move face->mesh index to scene?
    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, &system->face_to_mesh_index);

    return THERMAL_EQUATION_OK;
  }

  // hit side detection
  /*math::vec3 normale = math::triangle_normal(*ray_caster_task->hit_face[r]);
  float side = dot(normale, source.direction);
  int mesh_idx = face2mesh(system, ray_caster_task->hit_face[r] - system->emission_scene.faces);
  const subject::material_t& material = mesh_material(system->scene, mesh_idx);
  task->absorption[mesh_idx] += (side < 0 ? material.front.absorbance : material.rear.absorbance) * ray_power;*/

  /**
  *  @brief Calculates thermal flow for a given scene.
  */
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    // Prepare emission task
    task_reset(system->emission_task->rays);
    system->emission_task->mesh_temperatures = task->temperatures;
    system->emission_task->mesh_emission = task->emission;

    int r = 0;
    for (int e = 0; e != system->params.n_emitters; ++e)
    {
      emission::system_t* emitter = system->params.emitters[e];
      if ((r = system_calculate(emitter, system->emission_task)) < 0)
        return r;
    }  

    if ((r = system_cast(system->params.ray_caster, system->emission_task->rays)) < 0)
      return r;

    const emission::task_t* emission_task = system->emission_task;
    const ray_caster::task_t* ray_caster_task = emission_task->rays;
    const int n_rays = ray_caster_task->n_tasks;
    for (int r = 0; r != n_rays; ++r)
    {
      if (ray_caster_task->hit_face[r])
      {
        const int hit_face_idx = ray_caster_task->hit_face[r] - system->scene->faces;
        const int hit_mesh_idx = system->face_to_mesh_index[hit_face_idx];
        const float ray_power = emission_task->power[r];
        task->absorption[hit_mesh_idx] += ray_power;
      }
    }

    return THERMAL_EQUATION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system, thermal_equation::task_t* task))&calculate,
  };

  thermal_equation::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }

}
