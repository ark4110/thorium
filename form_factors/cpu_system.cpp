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
 * This module contains CPU single-threaded implementation of form factors calculator.
 * Calculator is capable to work with CPU or GPU ray caster implementation.
 */

#include "cpu_system.h"
#include "../ray_caster/system.h"
#include "../emission/malley_cpu.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include "../math/ray.h"
#include <float.h>
#include <cmath>
#include <stdlib.h>
#include <cstring>

namespace cpu_form_factors
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : form_factors::system_t
  {
    subject::scene_t* scene;

    float* mesh_temperatures;
    float* mesh_emission;

    /// @brief Inverted index to find mesh from face idx.
    int* face_to_mesh;

    emission::system_t* emitter;
    ray_caster::system_t* caster;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, const form_factors::params_t* params)
  {
    system->scene = 0;

    system->mesh_temperatures = 0;
    system->mesh_emission = 0;
    
    system->face_to_mesh = 0;
    system->emitter = params->emitter;
    system->caster = params->caster;
    
    return FORM_FACTORS_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    free(system->mesh_temperatures);
    system->mesh_temperatures = 0;

    free(system->mesh_emission);
    system->mesh_emission = 0;

    free(system->face_to_mesh);
    system->face_to_mesh = 0;

    system->emitter = 0;
    system->caster = 0;

    return FORM_FACTORS_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;

    int n_meshes = scene->n_meshes;
    
    int r = 0;
    
    if ((r = system_set_scene(system->caster, scene)) < 0)
      return r;
    if ((r = system_prepare(system->caster)) < 0)
      return r;
    if ((r = emission::system_set_scene(system->emitter, system->scene)) < 0)
      return r;

    // @todo Move face->mesh index to scene?
    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, &system->face_to_mesh);

    system->mesh_emission = (float*)calloc(n_meshes, sizeof(float));
    system->mesh_temperatures = (float*)calloc(n_meshes, sizeof(float));
    for (int m = 0; m != n_meshes; ++m)
      system->mesh_temperatures[m] = 1000;

    return FORM_FACTORS_OK;
  }

  /// @brief Returns mesh for given face.
  int face2mesh(cpu_system_t* system, int face_idx)
  {
    return system->face_to_mesh[face_idx];
  }

  /**
   *  @brief Calculates form factors for given system.
   *
   *  System uses ray caster (@see init()) and given task for N rays and scene's meshes.
   */
  int calculate(cpu_system_t* system, form_factors::task_t* task)
  {
    emission::system_t* emitter = system->emitter;

    int r = 0;    

    emission::task_t* emission_task = emission::task_create();
    emission_task->mesh_emission = system->mesh_emission;
    emission_task->mesh_temperatures = system->mesh_temperatures;
    
    system_relocate_task(system->emitter, emission_task, 0);
    if ((r = emission::system_calculate(emitter, emission_task)) < 0)
    {
      task_free(emission_task);
      return r;
    }

    if ((r = system_relocate_task(system->caster, emission_task->rays, 0)) < 0)
      return r;
    if ((r = ray_caster::system_cast(system->caster, emission_task->rays)) < 0)
    {
      task_free(emission_task);
      return r;
    }
    if ((r = system_relocate_task(system->caster, emission_task->rays, -1)) < 0)
      return r;

    // now we have nearest intersection face for every ray in task (if intersection was occurred)
    // calculate form factors between meshes
    const int n_meshes = system->scene->n_meshes;
    memset(task->form_factors, 0, n_meshes * n_meshes * sizeof(float));

    ray_caster::task_t* ray_caster_task = emission_task->rays;
    
    int n_rays = emission_task->rays->n_tasks;
    int mesh_idx = 0;
    int face_idx = 0;
    int mesh_outgoing_rays = 0;

    for (int ray_idx = 0; ray_idx != n_rays && mesh_idx != n_meshes;)
    {      
      const subject::mesh_t& mesh = system->scene->meshes[mesh_idx];
      const ray_caster::face_t& face = system->scene->faces[mesh.first_idx + face_idx];
      const math::ray_t ray = ray_caster_task->ray[ray_idx];

      bool current_face = ray_origin_on_plane(ray, face);
      if (!current_face)
      {
        if (++face_idx == mesh.n_faces)
        {
          float mesh_ratio = 1.f / (float)mesh_outgoing_rays;
          for (int j = 0; j != n_meshes; ++j)
            task->form_factors[mesh_idx * n_meshes + j] *= mesh_ratio;

          ++mesh_idx;
          face_idx = 0;
          mesh_outgoing_rays = 0;
        }
        continue;
      }
      ++mesh_outgoing_rays;

      if (ray_caster_task->hit_face_idx[ray_idx] != -1)
      {
        int hit_face_idx = ray_caster_task->hit_face_idx[ray_idx];
        int hit_mesh_idx = face2mesh(system, hit_face_idx);
        task->form_factors[mesh_idx * n_meshes + hit_mesh_idx] += 1.f;
      }
      ++ray_idx;
    }
    float mesh_ratio = 1.f / (float)mesh_outgoing_rays;
    for (int j = 0; j != n_meshes; ++j)
      task->form_factors[mesh_idx * n_meshes + j] *= mesh_ratio;

    task_free(emission_task);
    
    return FORM_FACTORS_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const form_factors::system_methods_t methods =
  {
    (int(*)(form_factors::system_t* system, const form_factors::params_t*))&init,
    (int(*)(form_factors::system_t* system))&shutdown,
    (int(*)(form_factors::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(form_factors::system_t* system, form_factors::task_t* task))&calculate,
  };

  form_factors::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
  
}
