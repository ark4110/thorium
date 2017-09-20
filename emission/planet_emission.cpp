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

#include "planet_emission.h"
#include "parallel_rays_cpu.h"
#include <stdlib.h>
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <math.h>

namespace planet_emission
{
  struct system_t : emission::system_t
  {
    params_t params;
    
    parallel_rays_emission_cpu::params_t element_params;
    emission::system_t* element_emitter;

    subject::scene_t* scene;    
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(system_t* system, const params_t* params)
  {
    system->params = *params;
    system->scene = 0;

    int n_elements = (2 * params->n_subdisivion - 1) * (2 * params->n_subdisivion - 1);
    system->element_params.n_sources = 1 + n_elements; // +1 for solar flux
    system->element_params.sources = (parallel_rays_emission_cpu::parallel_rays_source_t*)calloc(system->element_params.n_sources, sizeof(parallel_rays_emission_cpu::parallel_rays_source_t));
    system->element_emitter = system_create(params->parallel_rays_emitter_type, &system->element_params);

    return EMISSION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(system_t* system)
  {
    int r = 0;
    if ((r = system_shutdown(system->element_emitter)) < 0)
      return r;

    system_free(system->element_emitter);
    free(system->element_params.sources);
    system->element_params = {};

    system->scene = 0;
    return EMISSION_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(system_t* system, subject::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    system->scene = scene;

    int r = 0;
    if ((r = system_set_scene(system->element_emitter, scene)) < 0)
      return r;

    return EMISSION_OK;
  }

  /**
    @detail Based on http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020024.pdf
    An Earth albedo model.
    A mathematical model for the radiant energy input to an oribiting spacecraft
    due to the diffuse reflectance of solar radiation from the Earth below.
    Thomas W. Flatley and Wendy A. Moore, 1994
  */
  int calculate(system_t* system, emission::task_t* task)
  {
    int r = 0;

    const params_t& params = system->params;
    int n_elements = (2 * params.n_subdisivion - 1) * (2 * params.n_subdisivion - 1);
    int n_element_rays =  std::max<int>(1, params.n_rays / n_elements);

    float inverse_na = 1.f / n_elements;
    
    const float Re = params.planet_radius;
    const float iRe = 1.f / Re;

    ballistics::prediction_t prediction;
    if ((r = predict_orbit(params.predictor, &prediction)) < 0)
      return r;

    float total_flux = 0;
    parallel_rays_emission_cpu::parallel_rays_source_t& solar_source = system->element_params.sources[n_elements];
    if (params.n_solar_rays > 0 && params.solar_flux > 0 && !prediction.eclispsed)
    { 
      solar_source.n_rays = params.n_solar_rays;
      solar_source.power = params.solar_flux;
      solar_source.direction = math::normalize(-prediction.sun_position);
      total_flux += params.solar_flux;
    }
    else
    {
      solar_source.n_rays = 0;
      solar_source.power = 0;
    }

    // Distance to earth
    float R = math::norm(prediction.earth_position);
    float R2Re2 = (R * R- Re * Re);

    // Local 1-2-3 coordinate system and transformation from attitude coords to local.
    math::mat33 local_transform = math::rotate_towards(math::normalize(prediction.earth_position), math::make_vec3(0, 0, -1));
    math::mat33 global_transform = math::rotate_towards(math::make_vec3(0, 0, -1), math::normalize(prediction.earth_position));

    // s - unit vector from Earth to Sun in local 1-2-3 coordinate system.
    math::vec3 s = math::normalize(local_transform * (prediction.sun_position - prediction.earth_position));

    // Field of satellite view angle
    float cos_alpha_max = sqrtf(1 - Re * Re / R / R);
    float Amax = float(M_2PI) * (1 - cos_alpha_max);
    float A1 = Amax * inverse_na;
    float dFlux = params.solar_flux * params.planet_albedo * A1 / float(M_PI);

    const float sigma = 5.670400e-8f;
    float ePower = sigma * params.planet_temperature * params.planet_temperature * params.planet_temperature * params.planet_temperature;
    float eFlux = ePower * A1 / float(M_PI);

    // Init with central element
    float factor0 = dot(s, math::make_vec3(0, 0, 1));
    total_flux += factor0 > 0 ? factor0 * dFlux : 0;

    int n_element = 0;
    parallel_rays_emission_cpu::parallel_rays_source_t& source = system->element_params.sources[n_element];
    source.n_rays = 0;
    if (factor0 > 0)
    {
      source.n_rays = n_element_rays;
      source.power = factor0 * dFlux;
      source.direction = global_transform * math::make_vec3(0, 0, 1);
    }
    ++n_element;

    int k = 0;
    for (int i = 2; i <= params.n_subdisivion; ++i)
    {
      float ri_square = 4.f * (i - 1) * (i - 1) + 1;
      float cos_alpha = (n_elements - ri_square + ri_square * cos_alpha_max) * inverse_na; // circle center angle
      float cos_gamma = cos_alpha;
      float base = R * cos_gamma;
      float deter = sqrtf(base * base - R2Re2);

      // base = -R * u3, gamma may be > pi/2, so cos_gamma may be negative
      float d = base < 0 ? base + deter : base - deter; // select nearest point

      float sin_gamma = sinf(acosf(cos_gamma)); // safe to acos because quadrant sign is not a problem (< pi)
      int m = 8 * (i - 1); // circle subdivisions count
      float theta = float(M_2PI) / m;
      for (int j = 1; j <= m; ++j, ++k, ++n_element)
      {
        float delta = theta * j - theta / 2.f;
        float cos_delta = cosf(delta);
        float sin_delta = sinf(delta); // don't use sqrtf(1 - cos(delta)^2). quadrant sign is not a problem

        // u - unit vector from satellite to earth surface element dA in satellite 1-2-3 coordinate system
        math::vec3 u = math::make_vec3(sin_gamma * cos_delta, sin_gamma * sin_delta, -cos_gamma);

        // ne - unit vector of normal of earth surface element dA 
        math::vec3 ne = normalize(iRe * (math::make_vec3(0, 0, R) + d * u)); // Believe in compiler optimization

        float dFactor = math::dot(ne, s);

        parallel_rays_emission_cpu::parallel_rays_source_t& source = system->element_params.sources[n_element];
        source.n_rays = 0;
        // On sun lit territory
        if (dFactor > 0 || eFlux > 0)
        {
          source.n_rays = n_element_rays;
          float flux = (dFactor > 0 ? dFactor : 0.f) * dFlux + eFlux;
          total_flux += flux;
          source.power = flux;
          source.direction = global_transform * (-u);
        }
      }
    }

    if ((r = system_relocate_task(system->element_emitter, task, 0)) < 0)
      return r;
    if ((r = system_calculate(system->element_emitter, task)) < 0)
      return r;

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
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
