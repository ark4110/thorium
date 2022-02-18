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
// along with thorium.  If not, see <http://www.gnu.org/licenses/>.

/**
* This module contains CPU single-threaded implementation of emission calculator.
* Calculator is capable of work with CPU or GPU ray caster implementation.
*/

#include "malley_cpu.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include "../math/ray.h"
#include <assert.h>
#include <random>
#include <cmath>
#include <stdlib.h>
#include <cstring>

namespace malley_cpu
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : emission::system_t
  {
    params_t params;
    
    subject::scene_t* scene;

    float* face_weights;

    // Mersenne's twister uniformly distributed [0, 1) generators.
    std::mt19937 TPGenA;
    std::mt19937 TPGenB;

    // [0, 1)
    std::uniform_real_distribution<float> Distr_0_1;

    math::ray_malley_emission* malley_emission;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, const malley_cpu::params_t* params)
  {
    system->params = *params;
    
    system->scene = 0;    
    system->face_weights = 0;

    system->TPGenA = std::mt19937(1);
    system->TPGenB = std::mt19937(2);
    system->Distr_0_1 = std::uniform_real_distribution<float>(0, 1);
    system->malley_emission = math::ray_malley_emission_create();

    return EMISSION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    free(system->face_weights);
    system->face_weights = 0;

    math::ray_malley_emission_free(system->malley_emission);
    system->malley_emission = 0;

    return EMISSION_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    if (scene->n_meshes == 0)
      return -EMISSION_ERROR;

    if (scene->n_materials == 0)
      return -EMISSION_ERROR;

    system->scene = scene;

    system->face_weights = (float*)realloc(system->face_weights, 2 * scene->n_faces * sizeof(float));
    memset(system->face_weights, 0, 2 * scene->n_faces * sizeof(float)); // reset all values to zero to avoid arithmetic exceptions in future

    return EMISSION_OK;
  }

  void calculate_weights(cpu_system_t* system, const emission::task_t* task)
  {
    const float* temperatures = task->mesh_temperatures;
    const float* face_areas = system->scene->face_areas;
    subject::scene_t* scene = system->scene;

    // Don't reset weights to zero because we don't care about orphan faces without mesh.
    float* weights = system->face_weights;

    float total = 0;
    for (int m = 0; m != scene->n_meshes; ++m)
    {
      const subject::material_t& material = mesh_material(scene, m);
      const float T = temperatures[m];
      const float density = T; // some empirical distribution function of temperature
      const float front_density = density * material.front.emission.emissivity;
      const float rear_density = density * material.rear.emission.emissivity;

      const subject::mesh_t& mesh = scene->meshes[m];
      for (int f = mesh.first_idx; f != mesh.first_idx + mesh.n_faces; ++f)
      {
        const float front = front_density * face_areas[f];
        const float rear = rear_density * face_areas[f];
        total += front + rear;
        weights[f * 2] = front;
        weights[f * 2 + 1] = rear;
      }
    }

    float factor = 1.f / total;

    const int n_weights = scene->n_faces * 2; // there are at most 2 * n_faces weights (some may be unused)    
    for (int f = 0; f != n_weights; ++f) // Using loop over all faces (not mesh-by-mesh) for better performance
    {
      weights[f] *= factor;
    }
  }

  int emitted_front(cpu_system_t* system, int face_idx)
  {
    const float weight = system->face_weights[2 * face_idx];
    return weight > 0 ? std::max<int>(1, (int)(system->params.n_rays * weight)) : 0;
  }

  int emitted_rear(cpu_system_t* system, int face_idx)
  {
    const float weight = system->face_weights[2 * face_idx + 1];
    return weight > 0 ? std::max<int>(1, (int)(system->params.n_rays * weight)) : 0;
  }

  int calculate_n_rays(cpu_system_t* system, emission::task_t* task)
  {
    int n_faces = system->scene->n_faces;
    int result = 0;
    for (int f = 0; f != n_faces; ++f)
    {
      const int face_rays_front = emitted_front(system, f);
      const int face_rays_rear = emitted_rear(system, f);
      result += face_rays_front + face_rays_rear;
    }
    return result;
  }

  /// @brief Generates uniformly distributed points on triangle.
  math::vec3 pick_face_point(cpu_system_t* system, const math::face_t& face)
  {
    float a = system->Distr_0_1(system->TPGenA);
    float b = system->Distr_0_1(system->TPGenB);
    if (a + b > 1)
    {
      a = 1.f - a;
      b = 1.f - b;
    }

    math::vec3 v0 = face.points[1] - face.points[0];
    math::vec3 v1 = face.points[2] - face.points[0];
    return face.points[0] + a * v0 + b * v1;
  }

  const float sigma = 5.670400e-8f;

  void update_mesh_emission(const subject::scene_t* scene, emission::task_t* task)
  {
    for (int m = 0; m != scene->n_meshes; ++m)
    {
      const subject::material_t& material = mesh_material(scene, m);
      double mesh_area = scene->mesh_areas[m];
      double T = task->mesh_temperatures[m];
      double power_density = (T * T * T * T) * sigma;
      double emission = power_density * mesh_area * (material.front.emission.emissivity + material.rear.emission.emissivity);
      assert(!std::isnan(emission) && !std::isinf(emission) && "bad value detected");
      task->mesh_emission[m] = (subject::mesh_emission_t)(emission + task->mesh_emission[m]);
    }
  }

  int calculate(cpu_system_t* system, emission::task_t* task)
  {
    int r;
    if ((r = subject::scene_relocate(system->scene, -1)) < 0)
      return r;

    // Calculate total amount of rays to emmit.
    calculate_weights(system, task);
    int n_real_rays = calculate_n_rays(system, task);

    if (n_real_rays == 0)
      return EMISSION_OK;

    // Prepare caster rays storage.    
    int n_input_rays = task_grow(task, n_real_rays);
    ray_caster::task_t* ray_caster_task = task->rays;

    // Just syntactic sugar
    subject::scene_t* scene = system->scene;
    const int n_meshes = system->scene->n_meshes;

    int n_ray = 0;

    if (task->mesh_rays_emitted)
      memset(task->mesh_rays_emitted, 0, n_meshes * sizeof(int));

    for (int m = 0; m != n_meshes; ++m)
    {
      const subject::mesh_t& mesh = scene->meshes[m];
      const subject::material_t& material = mesh_material(scene, m);

      // Mesh radiance properties.
      const float T = task->mesh_temperatures[m];
      const float power_density = sigma * (T * T * T * T);
      const float front_density = power_density * material.front.emission.emissivity;
      const float rear_density = power_density * material.rear.emission.emissivity;

      // Emmit rays from each mesh face
      for (int f = mesh.first_idx; f != mesh.first_idx + mesh.n_faces; ++f)
      { 
        const float front_emission = front_density * system->scene->face_areas[f];
        const float rear_emission = rear_density * system->scene->face_areas[f];

        const int n_face_rays_front = emitted_front(system, f);
        const int n_face_rays_rear = emitted_rear(system, f);

        if (task->mesh_rays_emitted)
          task->mesh_rays_emitted[m] += (n_face_rays_front + n_face_rays_rear);

        const ray_caster::face_t& face = scene->faces[f];

        // Store rotation for a given face (from Z axis towards face's normal).
        math::mat33 rotation = pick_face_rotation(face, math::make_vec3(0, 0, 1));
        
        for (int j = 0; j != n_face_rays_front && n_ray < n_real_rays; ++j, ++n_ray)
        {
          const float ray_power = front_emission / n_face_rays_front;
          task->rays->power[n_input_rays + n_ray] = ray_power;

          // Randomly generated ray's origin on the face
          math::vec3 origin = pick_face_point(system, face);

          // Pick direction from cosine-weighted distribution
          math::vec3 malley = ray_malley_emmit(system->malley_emission);

          // Rotate ray towards face's normal
          math::vec3 relative_dist = rotation * malley;

          ray_caster_task->ray[n_input_rays + n_ray] = { origin + relative_dist * 0.0001f, relative_dist };
        }

        // Duplicate "front" cycle to avoid branches.
        for (int j = 0; j != n_face_rays_rear && n_ray < n_real_rays; ++j, ++n_ray)
        {
          const float ray_power = rear_emission / n_face_rays_rear;
          task->rays->power[n_input_rays + n_ray] = ray_power;
          
          math::vec3 origin = pick_face_point(system, face);
          math::vec3 malley = ray_malley_emmit(system->malley_emission);

          // Only difference from front face is opposite direction.
          malley.z = -malley.z;
          math::vec3 relative_dist = rotation * malley;
          ray_caster_task->ray[n_input_rays + n_ray] = { origin + relative_dist * 0.0001f, relative_dist };
        }
      }
    }

    update_mesh_emission(system->scene, task);

    return EMISSION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const emission::system_methods_t methods =
  {
    (int(*)(emission::system_t* system, const emission::params_t*))&init,
    (int(*)(emission::system_t* system))&shutdown,
    (int(*)(emission::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(emission::system_t* system, emission::task_t* task))&calculate,
    &emission::native_host_task_relocation
  };

  emission::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }

  int emitted_front(emission::system_t* system, int face_idx)
  {
    return emitted_front((cpu_system_t*)system, face_idx);
  }

  int emitted_rear(emission::system_t* system, int face_idx)
  {
    return emitted_rear((cpu_system_t*)system, face_idx);
  }
}
