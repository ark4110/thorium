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
* This module contains CPU single-threaded implementation of emission calculator.
* Calculator is capable of work with CPU or GPU ray caster implementation.
*/

#include "parallel_rays_cpu.h"
#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <float.h>
#include <random>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace parallel_rays_emission_cpu
{
  struct cpu_system_t : emission::system_t
  { 
    params_t params;

    subject::scene_t* scene;    
    math::sphere_t bsphere;

    // Mersenne's twister uniformly distributed [0, 1) generators.
    std::mt19937 TPGenR;
    std::mt19937 TPGenTheta;

    std::uniform_real_distribution<float> Distr_R;
    std::uniform_real_distribution<float> Distr_Theta;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, const params_t* params)
  {
    system->params = *params;

    system->scene = 0;    

    system->TPGenR = std::mt19937(1);
    system->TPGenTheta = std::mt19937(2);
    system->Distr_R = std::uniform_real_distribution<float>(0, 1);
    system->Distr_Theta = std::uniform_real_distribution<float>(0, 2 * (float)M_PI);
    return EMISSION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;
    return EMISSION_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    system->scene = scene;
    system->bsphere = math::triangles_bsphere(scene->faces, scene->n_faces);

    return EMISSION_OK;
  }

  int calculate(cpu_system_t* system, emission::task_t* task)
  {
    for (int source_idx = 0; source_idx != system->params.n_sources; ++source_idx)
    {
      parallel_rays_source_t& source = system->params.sources[source_idx];
      int n_rays = source.n_rays;
      int n_input_rays = task_grow(task, n_rays);

      float radius = system->bsphere.radius * scene_bsphere_emission_scale;
      float offset = 1.3f + radius;
      float ray_power = source.power * (float(M_PI) * radius * radius) / n_rays;

      math::mat33 rotation = math::rotate_towards(math::make_vec3(0, 0, 1), source.direction);

      for (int r = 0; r != n_rays; ++r)
      {
        float radius_dice = radius * sqrtf(system->Distr_R(system->TPGenR));
        float theta_dice = system->Distr_Theta(system->TPGenTheta);
        float x = radius_dice * cosf(theta_dice);
        float y = radius_dice * sinf(theta_dice);
        math::vec3 origin = rotation * math::make_vec3(x, y, -offset) + system->bsphere.center;
        math::vec3 direction = source.direction;

        task->rays->ray[n_input_rays + r] = { origin, direction };
        task->power[n_input_rays + r] = ray_power;
      }
    }

    return EMISSION_OK;
  }

  const emission::system_methods_t methods =
  {
    (int(*)(emission::system_t* system, const emission::params_t*))&init,
    (int(*)(emission::system_t* system))&shutdown,
    (int(*)(emission::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(emission::system_t* system, emission::task_t* task))&calculate,
  };

  emission::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
