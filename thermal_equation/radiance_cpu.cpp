// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
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

#include <chrono>

namespace radiance_equation
{
  struct cpu_system_t : thermal_equation::system_t
  {    
    params_t params;

    subject::scene_t* scene;

    emission::task_t* emission_task;
    double* mesh_absorption;
    
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
    system->mesh_absorption = 0;

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

    ray_malley_emission_free(system->malley_emission);
    system->malley_emission = 0;

    free(system->mesh_absorption);
    system->mesh_absorption = 0;

    return THERMAL_EQUATION_OK;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;

    int r = 0;
    if ((r = system_set_scene(system->params.ray_caster, scene)) < 0)
      return r;

    if ((r = system_update(system->params.ray_caster)) < 0)
      return r;

    for (int e = 0; e != system->params.n_emitters; ++e)
    if ((r = system_set_scene(system->params.emitters[e], scene)) < 0)
      return r;

    emission::task_free(system->emission_task);
    system->emission_task = emission::task_create();
    system->mesh_absorption = (double*)realloc(system->mesh_absorption, sizeof(double) * scene->n_meshes);
    
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
    task_next_add(system->emission_task, ray, power);
  }

  /**
  *  @brief Calculates thermal flow for a given scene.
  */
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    // Prepare emission task
    emission::task_t* emission_task = system->emission_task;
    task_next_gen(emission_task->rays);

    if (system->params.stats)
      *system->params.stats = stats_t();

    emission_task->mesh_temperatures = task->temperatures;
    emission_task->mesh_emission = task->emission;

    auto start_ptime = std::chrono::high_resolution_clock().now();

    int r = 0;
    for (int e = 0; e != system->params.n_emitters; ++e)
    {
      emission::system_t* emitter = system->params.emitters[e];
      if ((r = system_calculate(emitter, system->emission_task)) < 0)
        return r;
    }
    auto emission_ptime = std::chrono::high_resolution_clock().now();
    
    // @todo Is nothing to do ok?
    if (system->emission_task->rays == 0)
      return THERMAL_EQUATION_OK;

    if ((r = system_relocate_task(system->params.ray_caster, emission_task->rays, 0)) < 0)
      return r;
    if ((r = system_cast(system->params.ray_caster, emission_task->rays)) < 0)
      return r;
    if ((r = system_relocate_task(system->params.ray_caster, emission_task->rays, -1)) < 0)
      return r;
    auto ray_caster_ptime = std::chrono::high_resolution_clock().now();

    for (int i = 0; i != system->scene->n_meshes; ++i)
      system->mesh_absorption[i] = task->absorption[i];
    
    const ray_caster::task_t* ray_caster_task = emission_task->rays;
    const int n_rays = ray_caster_task->n_tasks;
    for (int r = 0; r != n_rays; ++r)
    {
      if (ray_caster_task->hit_face_idx[r] == -1)
      {
        if (system->params.stats)
        {
          const float ray_power = emission_task->rays->power[r];
          ++system->params.stats->quants.named.n_missed;
          system->params.stats->powers.named.e_missed += ray_power;
        }
      }
      else
      {
        const int hit_face_idx = ray_caster_task->hit_face_idx[r];
        const int hit_mesh_idx = system->scene->face_to_mesh[hit_face_idx];

        const math::ray_t& ray = ray_caster_task->ray[r];
        const math::triangle_t& face = system->scene->faces[hit_face_idx];

        const subject::material_t& material = mesh_material(system->scene, hit_mesh_idx);
        bool hit_front = ray_check_hit_front(ray, face);
        const subject::optical_properties_t& optical = hit_front ? material.front.optical : material.rear.optical;

        const float ray_power = emission_task->rays->power[r];

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
          system->mesh_absorption[hit_mesh_idx] += ray_power;
        }
        break;

        case hit_reflect:
        {
          math::vec3 hit_point = ray_point(ray, ray_caster_task->hit_distance[r]);
          math::ray_t reflected = ray_reflect(ray, face, hit_point);
          postpone(system, reflected, ray_power);
        }
        break;

        case hit_diffuse:
        {
          math::vec3 hit_point = ray_point(ray, ray_caster_task->hit_distance[r]);
          math::ray_t reflected = ray_malley_emmit(system->malley_emission, face, hit_point, hit_front);
          postpone(system, reflected, ray_power);
        }
        break;

        case hit_transmit:
        {
          math::vec3 hit_point = ray_point(ray, ray_caster_task->hit_distance[r]);
          math::ray_t transmitted = ray_transmit(ray, hit_point);
          postpone(system, transmitted, ray_power);
        }
        break;

        default:
          break;
        }
      }
    }

    for (int i = 0; i != system->scene->n_meshes; ++i)
      task->absorption[i] = (float)system->mesh_absorption[i];

    auto processing_ptime = std::chrono::high_resolution_clock().now();

    if (system->params.stats)
    {
      system->params.stats->quants.named.n_postponed = emission_task->rays->next_gen.n_tasks;

      auto emission_time = std::chrono::duration_cast<std::chrono::microseconds>(emission_ptime - start_ptime);
      auto ray_caster_time = std::chrono::duration_cast<std::chrono::microseconds>(ray_caster_ptime - emission_ptime);
      auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(processing_ptime - ray_caster_ptime);

      system->params.stats->times.named.t_emission += (int)emission_time.count();
      system->params.stats->times.named.t_ray_casting += (int)ray_caster_time.count();
      system->params.stats->times.named.t_processing += (int)processing_time.count();

      ++system->params.stats->n_steps;
    }

    return THERMAL_EQUATION_OK;
  }

  int update(cpu_system_t* system)
  {
    return ray_caster::system_update(system->params.ray_caster);
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system))&update,
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

    /*fprintf(stderr, "RADIANCE\tRays power: postponed %.2Ef, absorbed %.2Ef, reflected %.2Ef, diffused %.2Ef, transmited %.2Ef, droped %.2Ef, missed %.2Ef\n"
      , stats.powers.named.e_postponed
      , stats.powers.named.e_absorbed
      , stats.powers.named.e_reflected
      , stats.powers.named.e_diffused
      , stats.powers.named.e_transmited
      , stats.powers.named.e_droped
      , stats.powers.named.e_missed
    );*/

    {
      int emission = (stats.times.named.t_emission / stats.n_steps) / 1000;
      int ray_casting = (stats.times.named.t_ray_casting / stats.n_steps) / 1000;
      int processing = (stats.times.named.t_processing / stats.n_steps) / 1000;
      fprintf(stderr, "RADIANCE\tTimings: emission %dms ray-tracing %dms processing %dms\n"
        , emission
        , ray_casting
        , processing
        );
    }
  }
}
