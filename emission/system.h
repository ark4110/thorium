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
 * This module contains basic types to represent a scene emission calculation.
 * Scene emission generate ray-caster system rays for radiosity calculation.
 * Physical interpretation of results is out of scope of this library.
 * Module also contains base type (system_t) for emission calculation with table of virtual methods.
 */

#pragma once

#include "../math/types.h"
#include "../ray_caster/system.h"
#include "../subject/system.h"


/// @todo Rename "emission" to "radiance"
namespace emission
{
  /**
   * @brief Task representation for given scene (@see task_create).
   *
   * Task result consists of N rays (as input for ray_caster) and rays powers.
   */
  struct task_t
  {
    /**
    @brief Input meshes temperatures, K.
    @todo In general case emission don't need mesh temperature. Move it to black body emission?
    */
    const float* mesh_temperatures;

    /**
      @brief Ray caster task contains calculation output.
      @detail Result may be allocated in case of failure.
      @detail Owned by emission system.
      @note Will reallocate and add additional rays to input task.
      @note Actual rays count can differ from requested n_rays count.
      */
    ray_caster::task_t* rays;

    /**
      @brief Amount of energy mesh loose during emission, W.
      @todo In general case emission is not related to objects. Move it to black body emission?
      @detail Owned by client.
    */
    float* mesh_emission;

    /**
      @brief Rays count emitted by mesh.
      @detail Owned by client.
    */
    int* mesh_rays_emitted;
  };

  task_t* task_create();
  void task_free(task_t* task);
  
  /// @brief Add more rays to task result.
  /// @return Previous rays count.
  int task_grow(task_t* task, int n_rays);
  int task_reserve(task_t* task, int n_rays);
  void task_next_add(task_t* task, const ray_caster::ray_t& ray, float power);

  /**
   * @brief Emission system.   
   */
  struct system_t
  {
    /// @brief virtual methods table.
    const struct system_methods_t* methods;
  };

  struct params_t
  {
  };

  #define EMISSION_OK    0
  #define EMISSION_ERROR 200
  #define EMISSION_INVALID_TASK_ALLOCATION 201

  /**
   *   @brief Virtual methods table for calculator functionality.
   *
   *   Every concrete calculator should implement these methods.
   *   C-interface functions (system_* functions, see below) just allocate memory and call these virtual methods.
   */
  struct system_methods_t
  { 
    /// @brief Initializes system with given ray caster after creation.
    int(*init)(system_t* system, const params_t* params);

    /// @brief Shutdowns calculator system prior to free memory.
    int(*shutdown)(system_t* system);

    /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
    /// @todo Scene is not required in general case.
    int(*set_scene)(system_t* system, subject::scene_t* scene);

    /// @brief Calculates weights, generate emission rays and casts rays for given system.
    int(*calculate)(system_t* system, task_t* task);

    /**
      @detail -1 for host device, 0 for system native device.
      @detail Right now rays power are always on host. I.e. task relocation is equal to ray_caster task relocation.
      @todo Relocate rays power. Consider moving powers to ray caster task as additional meta.
    */
    int(*relocate_task)(system_t* system, task_t* task, int device);
  };

  #define EMISSION_MALLEY_CPU 1
  #define EMISSION_PARALLEL_RAYS_CPU 2
  #define EMISSION_MALLEY_CUDA 3
  #define EMISSION_PARALLEL_RAYS_CUDA 4
  #define EMISSION_PLANET_CPU 5

  /**
   * @brief Factory method for calculator creation.
   *
   * Creates emission calculator system with given calculator type and raycaster.
   * @note Only CPU calculator type (type = 1) is supported, @see ../cpuEmission/.
   * @note init() system on creation.
   */
  system_t* system_create(int type, const params_t* params);

  /// Here go C-interface wrappers to call system_t's virtual methods.

  /// @note shutdown() system on destruction.
  void system_free(system_t* system);
  int system_init(system_t* system, const params_t* params);
  int system_shutdown(system_t* system);
  int system_set_scene(system_t* system, subject::scene_t* scene);
  int system_calculate(system_t* system, task_t* task);
  int system_relocate_task(system_t* system, task_t* task, int device);

  // Relocation for CPU based systems.
  int native_host_task_relocation(emission::system_t*, emission::task_t* task, int device);
}
