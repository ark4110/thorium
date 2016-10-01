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
#include "../math/operations.h"
#include "../math/ray.h"
#include <float.h>
#include <cmath>
#include <random>
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
    
    int postponed_free;
    int postponed_alloc;
    math::ray_t* postponed_rays;
    float* postponed_powers;

    int* face_to_mesh_index;

    std::mt19937 HitGen;        
    std::uniform_real_distribution<float> Distr_0_1;

    math::ray_malley_emission* malley_emission;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, params_t* params)
  {    
    system->params = *params;

    system->scene = 0;
    system->emission_task = 0;
    system->postponed_free = 0;
    system->postponed_alloc = 0;
    system->postponed_rays = 0;
    system->postponed_powers = 0;
    system->face_to_mesh_index = 0;

    system->HitGen = std::mt19937(1);    
    system->Distr_0_1 = std::uniform_real_distribution<float>(0, 1);
    system->malley_emission = math::ray_malley_emission_create();

    return THERMAL_EQUATION_OK;
  }

  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    emission::task_free(system->emission_task);
    system->emission_task = 0;

    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;

    ray_malley_emission_free(system->malley_emission);
    system->malley_emission = 0;

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

  enum hit_decision
  {
    hit_absorb = 0,
    hit_reflect = 1,
    hit_diffuse = 2,
    hit_transmit = 3,
    hit_drop = 4
  };

  hit_decision make_hit_descision(cpu_system_t* system, const subject::optical_properties_t& optical)
  {
    float hit_dice = system->Distr_0_1(system->HitGen);

    // Lets believe in compiler optimization...
    if (hit_dice < optical.absorbance)
      return hit_absorb;
    else if (hit_dice < optical.absorbance + optical.specular_reflectance)
      return hit_reflect;
    else if (hit_dice < optical.absorbance + optical.specular_reflectance + optical.diffuse_reflectance)
      return hit_diffuse;
    else if (hit_dice < optical.absorbance + optical.specular_reflectance + optical.diffuse_reflectance + optical.transmittance)
      return hit_transmit;
    else
      return hit_drop;
  }

  void postpone(cpu_system_t* system, math::ray_t ray, float power)
  {
    if (system->postponed_free == 0)
    { 
      int grow = std::max<int>(1024, system->postponed_alloc);
      system->postponed_rays = (math::ray_t*)realloc(system->postponed_rays, (system->postponed_alloc + grow) * sizeof(math::ray_t));
      system->postponed_powers = (float*)realloc(system->postponed_powers, (system->postponed_alloc + grow) * sizeof(float));
      system->postponed_free = grow;
    }

    system->postponed_rays[system->postponed_alloc] = ray;
    system->postponed_powers[system->postponed_alloc] = power;
    ++system->postponed_alloc;
    --system->postponed_free;
  }

  /**
  *  @brief Calculates thermal flow for a given scene.
  */
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    // Prepare emission task
    emission::task_t* emission_task = system->emission_task;

    if (emission_task->rays)
    {
      // @todo Simplify code/logic and optimize: looks like there are a lot of false reallocs to zero.
      int n_allocated = task_reset(emission_task->rays);

      std::swap(emission_task->rays->ray, system->postponed_rays);
      std::swap(emission_task->power, system->postponed_powers);
      emission_task->rays->n_tasks = system->postponed_alloc;
      task_realloc(emission_task->rays, system->postponed_alloc);
      system->postponed_alloc = 0;
      system->postponed_free = n_allocated;
    }

    if (system->params.stats)
      *system->params.stats = stats_t();

    emission_task->mesh_temperatures = task->temperatures;
    emission_task->mesh_emission = task->emission;

    int n_rays_before_emission = emission_task->rays ? emission_task->rays->n_tasks : 0;
    int r = 0;
    for (int e = 0; e != system->params.n_emitters; ++e)
    {
      emission::system_t* emitter = system->params.emitters[e];
      if ((r = system_calculate(emitter, system->emission_task)) < 0)
        return r;
    }
    int n_rays_after_emission = emission_task->rays ? emission_task->rays->n_tasks : 0;

    // @todo Is nothing to do ok?
    if (system->emission_task->rays == 0)
      return THERMAL_EQUATION_OK;

    if ((r = system_cast(system->params.ray_caster, system->emission_task->rays)) < 0)
      return r;

    unsigned n_ray_counter[hit_drop + 1] = {};
    float ray_power_counter[hit_drop + 2] = {};

    const ray_caster::task_t* ray_caster_task = emission_task->rays;
    const int n_rays = ray_caster_task->n_tasks;
    for (int r = 0; r != n_rays; ++r)
    {
      if (!ray_caster_task->hit_face[r])
      {
        if (system->params.stats)
        {
          const float ray_power = emission_task->power[r];
          ++system->params.stats->quants.named.n_missed;
          system->params.stats->powers.named.e_missed += ray_power;
        }
      }
      else
      {
        const int hit_face_idx = ray_caster_task->hit_face[r] - system->scene->faces;
        const int hit_mesh_idx = system->face_to_mesh_index[hit_face_idx];

        const math::ray_t& ray = ray_caster_task->ray[r];
        const math::triangle_t& face = *ray_caster_task->hit_face[r];

        const subject::material_t& material = mesh_material(system->scene, hit_mesh_idx);
        const subject::optical_properties_t& optical = ray_check_hit_front(ray, face) ? material.front : material.rear;

        const float ray_power = emission_task->power[r];

        hit_decision decision = make_hit_descision(system, optical);
        if (system->params.stats)
        {
          ++system->params.stats->quants.raw[decision];
          system->params.stats->powers.raw[decision] += ray_power;
          if (decision != hit_absorb && decision != hit_drop)
            system->params.stats->powers.named.e_postponed += ray_power;
        }        

        switch (decision)
        {
        case hit_absorb:
        {
          task->absorption[hit_mesh_idx] += ray_power;
        }
        break;

        case hit_reflect:
        {
          math::ray_t reflected = ray_reflect(ray, face, ray_caster_task->hit_point[r]);
          postpone(system, reflected, ray_power);
        }
        break;

        case hit_diffuse:
        {
          math::ray_t reflected = ray_malley_emmit(system->malley_emission, face, ray_caster_task->hit_point[r]);
          postpone(system, reflected, ray_power);
        }
        break;

        case hit_transmit:
        {
          math::ray_t transmitted = ray_transmit(ray, ray_caster_task->hit_point[r]);
          postpone(system, transmitted, ray_power);
        }
        break;

        default:
          break;
        }
      }
    }

    if (system->params.stats)
      system->params.stats->quants.named.n_postponed = system->postponed_alloc;

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

  void print_stats(const stats_t& stats)
  {
    fprintf(stderr, "RADIANCE\tRays: postponed %d, absorbed %d, reflected %d, diffused %d, transmited %d, droped %d, missed %d\n"
      , stats.quants.named.n_postponed
      , stats.quants.named.n_absorbed
      , stats.quants.named.n_reflected
      , stats.quants.named.n_diffused
      , stats.quants.named.n_transmited
      , stats.quants.named.n_droped
      , stats.quants.named.n_missed
    );

    fprintf(stderr, "RADIANCE\tRays power: postponed %.2Ef, absorbed %.2Ef, reflected %.2Ef, diffused %.2Ef, transmited %.2Ef, droped %.2Ef, missed %.2Ef\n"
      , stats.powers.named.e_postponed
      , stats.powers.named.e_absorbed
      , stats.powers.named.e_reflected
      , stats.powers.named.e_diffused
      , stats.powers.named.e_transmited
      , stats.powers.named.e_droped
      , stats.powers.named.e_missed
    );
  }
}
