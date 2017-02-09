// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
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


#include "radiance_cuda.h"
#include "radiance_cuda.cuh"
#include "../math/triangle.h"
#include "../math/operations.h"
#include "../math/ray.h"
#include <float.h>
#include <cmath>
#include <random>
#include <limits>
#include <stdlib.h>
#include <assert.h>
#include <cstring>
#include <chrono>

#include <helper_cuda.h>
namespace {
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
}

#include <iostream>

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

namespace radiance_equation_cuda
{
  struct system_t : thermal_equation::system_t
  {
    params_t params;

    int n_tpb;
    int n_warp_size;
    int n_mtgp_blocks;

    context_t host_context;
    context_t* device_context;

    mtgp32_kernel_params *kernel_params;

    subject::scene_t* scene;

    emission::task_t* emission_task;
    emission::task_t* postponed_task;

    int* face_to_mesh_index;
  };

  void sync_context_to_device(system_t* system)
  {
    if (system->device_context == 0)
      checkCudaErrors(cudaMalloc((void**)&system->device_context, sizeof(context_t)));
    checkCudaErrors(cudaMemcpy(system->device_context, &system->host_context, sizeof(context_t), cudaMemcpyHostToDevice));
  }

  /// @brief Initializes system with given ray caster after creation.
  int init(system_t* system, params_t* params)
  {
    system->params = *params;

    system->host_context = {};
    system->device_context = 0;
    system->kernel_params = 0;

    system->emission_task = 0;
    system->postponed_task = 0;
    system->face_to_mesh_index = 0;

    const char* argv[] = { "" };
    int device_id = findCudaDevice(1, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_id));
    system->n_tpb = deviceProp.maxThreadsPerBlock;
    system->n_warp_size = deviceProp.warpSize;

    int n_mtgp_blocks = 64;
    system->n_mtgp_blocks = n_mtgp_blocks;

    context_t& host_context = system->host_context;
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenR, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenTheta, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenDescision, n_mtgp_blocks * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void**)&system->kernel_params, sizeof(mtgp32_kernel_params)));
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, system->kernel_params));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenR, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 3));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenTheta, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 4));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenDescision, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 4));

    return THERMAL_EQUATION_OK;
  }

  void contex_free_scene(context_t* ctx)
  {
    if (ctx->faces)
      cudaFree(ctx->faces);
    ctx->faces = 0;

    if (ctx->face_to_mesh_index)
      cudaFree(ctx->face_to_mesh_index);
    ctx->face_to_mesh_index = 0;

    if (ctx->materials)
      cudaFree(ctx->materials);
    ctx->materials = 0;

    if (ctx->mesh_materials)
      cudaFree(ctx->mesh_materials);
    ctx->mesh_materials = 0;
  }

  int shutdown(system_t* system)
  {
    system->scene = 0;

    contex_free_scene(&system->host_context);

    cudaFree(system->host_context.HSGenR);
    cudaFree(system->host_context.HSGenTheta);
    cudaFree(system->host_context.HSGenDescision);

    system->host_context = {};
    cudaFree(system->device_context);
    system->device_context = 0;

    cudaFree(system->kernel_params);
    system->kernel_params = 0;

    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;

    return THERMAL_EQUATION_OK;
  }

  float3 math2cuda(math::vec3 v)
  {
    return make_float3(v.x, v.y, v.z);
  }

  int set_scene(system_t* system, subject::scene_t* scene)
  {
    contex_free_scene(&system->host_context);

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

    emission::task_free(system->postponed_task);
    system->postponed_task = emission::task_create();

    context_t& host_context = system->host_context;

    // Prepare device faces
    cuda_math::prep_face_t* faces = (cuda_math::prep_face_t*)calloc(scene->n_faces, sizeof(cuda_math::prep_face_t));
    for (int i = 0; i != scene->n_faces; ++i)
      faces[i].normalized_normal = math2cuda(normalize(triangle_normal(scene->faces[i])));
    checkCudaErrors(cudaMalloc(&host_context.faces, sizeof(cuda_math::prep_face_t) * scene->n_faces));
    checkCudaErrors(cudaMemcpy(host_context.faces, faces, sizeof(cuda_math::prep_face_t) * scene->n_faces, cudaMemcpyHostToDevice));
    free(faces);

    // Init device face to mesh index
    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;
    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, &system->face_to_mesh_index);
    checkCudaErrors(cudaMalloc(&host_context.face_to_mesh_index, sizeof(int) * scene->n_faces));
    checkCudaErrors(cudaMemcpy(host_context.face_to_mesh_index, system->face_to_mesh_index, sizeof(int) * scene->n_faces, cudaMemcpyHostToDevice));
    
    // Init device materials
    checkCudaErrors(cudaMalloc((void**)&host_context.materials, sizeof(subject::material_t) * scene->n_materials));
    checkCudaErrors(cudaMemcpy(host_context.materials, scene->materials, sizeof(subject::material_t) * scene->n_materials, cudaMemcpyHostToDevice));

    // Init device mesh material index
    int* mesh_materials = (int*)calloc(scene->n_meshes, sizeof(int));
    for (int i = 0; i != scene->n_meshes; ++i)
      mesh_materials[i] = scene->meshes[i].material_idx;
    checkCudaErrors(cudaMalloc((void**)&host_context.mesh_materials, sizeof(int) * scene->n_meshes));
    checkCudaErrors(cudaMemcpy(host_context.mesh_materials, mesh_materials, sizeof(int) * scene->n_meshes, cudaMemcpyHostToDevice));
    free(mesh_materials);

    sync_context_to_device(system);

    return THERMAL_EQUATION_OK;
  }

  enum hit_decision
  {
    hit_absorb = 0,
    hit_reflect = 1,
    hit_diffuse = 2,
    hit_transmit = 3,
    hit_drop = 4,
    hit_missed = 5
  };

  __device__ hit_decision make_hit_descision(context_t* ctx, const subject::optical_properties_t& optical)
  {
    curandStateMtgp32* HSGenDescision = &ctx->HSGenDescision[blockIdx.x];
    float hit_dice = curand_uniform(HSGenDescision);

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

  __global__ void process_rays(context_t* ctx, int n_rays, const int* hit_face_idx, const float3* hit_point, ray_t* rays, char* hit_status)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += blockDim.x * gridDim.x)
    {
      const int face_idx = hit_face_idx[ray_idx];
      if (face_idx < 0)
      {
        hit_status[ray_idx] = hit_missed;
        continue;
      }

      const ray_t ray = rays[ray_idx];    
      const prep_face_t face = ctx->faces[face_idx];
      bool hit_front = cuda_math::ray_check_hit_front(ray, face);

      const int mesh_idx = ctx->face_to_mesh_index[face_idx];
      const subject::material_t& material = ctx->materials[ctx->mesh_materials[mesh_idx]];
      const subject::optical_properties_t& optical = hit_front ? material.front : material.rear;

      hit_decision decision = make_hit_descision(ctx, optical);
      const float3 point = hit_point[ray_idx];

      hit_status[ray_idx] = decision;
      switch (decision)
      {
      case hit_absorb:
        break;

      case hit_reflect:
        rays[ray_idx] = cuda_math::ray_reflect(ray, face, point);
        break;

      case hit_diffuse:
      {
        float3 malley_ray = malley_cuda::pick_uniform_ray(ctx);
        rays[ray_idx] = malley_cuda::ray_malley_emmit(malley_ray, face, hit_point[ray_idx]);
      }
      break;

      case hit_transmit:
        rays[ray_idx] = cuda_math::ray_transmit(ray, point);
        break;

      default:
        break;
      }
    }
  }

  void postpone(system_t* system, math::ray_t ray, float power)
  {
    task_add(system->postponed_task, ray, power);
  }

  /**
  *  @brief Calculates thermal flow for a given scene.
  */
  int calculate(system_t* system, thermal_equation::task_t* task)
  {
    // Prepare emission task
    std::swap(system->emission_task, system->postponed_task);
    task_reset(system->postponed_task);
    emission::task_t* emission_task = system->emission_task;

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

    if ((r = system_cast(system->params.ray_caster, system->emission_task->rays)) < 0)
      return r;
    auto ray_caster_ptime = std::chrono::high_resolution_clock().now();

    const ray_caster::task_t* ray_caster_task = emission_task->rays;
    const int n_rays = ray_caster_task->n_tasks;

    subject::scene_t* scene = system->scene;
    int* device_hit_face_idx = 0;
    checkCudaErrors(cudaMalloc(&device_hit_face_idx, sizeof(int) * n_rays));
    checkCudaErrors(cudaMemcpy(device_hit_face_idx, ray_caster_task->hit_face_idx, sizeof(int) * n_rays, cudaMemcpyHostToDevice));

    float3* device_hit_point = 0;
    checkCudaErrors(cudaMalloc(&device_hit_point, sizeof(float3) * n_rays));
    checkCudaErrors(cudaMemcpy(device_hit_point, ray_caster_task->hit_point, sizeof(float3) * n_rays, cudaMemcpyHostToDevice));

    ray_t* device_rays = 0;
    checkCudaErrors(cudaMalloc(&device_rays, sizeof(ray_t) * n_rays));
    checkCudaErrors(cudaMemcpy(device_rays, ray_caster_task->ray, sizeof(ray_t) * n_rays, cudaMemcpyHostToDevice));

    char* device_hit_status = 0;
    char* host_hit_status = (char*)malloc(n_rays);
    checkCudaErrors(cudaMalloc(&device_hit_status, sizeof(char) * n_rays));

    process_rays << <system->n_mtgp_blocks, 256 >> > (system->device_context
      , n_rays
      , device_hit_face_idx
      , device_hit_point
      , device_rays
      , device_hit_status);

    checkCudaErrors(cudaMemcpy(host_hit_status, device_hit_status, n_rays, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(ray_caster_task->ray, device_rays, sizeof(ray_t) * n_rays, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(device_hit_face_idx);
    cudaFree(device_hit_point);
    cudaFree(device_rays);
    cudaFree(device_hit_status);

    for (int r = 0; r != n_rays; ++r)
    {
      const int hit_face_idx = ray_caster_task->hit_face_idx[r];
      float ray_power = emission_task->power[r];

      if (hit_face_idx == -1)
      {
        if (system->params.stats)
        {
          ++system->params.stats->quants.named.n_missed;
          system->params.stats->powers.named.e_missed += ray_power;
        }
        continue;
      }

      const int hit_mesh_idx = system->face_to_mesh_index[hit_face_idx];

      assert(host_hit_status[r] <= hit_missed);

      hit_decision decision = (hit_decision)host_hit_status[r];
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
        task->absorption[hit_mesh_idx] += ray_power;
        break;

      case hit_reflect:
      case hit_diffuse:
      case hit_transmit:
        postpone(system, ray_caster_task->ray[r], ray_power);
      break;

      case hit_drop:
      case hit_missed:
      default:
        break;
      }
    }
    
    free(host_hit_status);

    auto processing_ptime = std::chrono::high_resolution_clock().now();

    if (system->params.stats)
    {
      system->params.stats->quants.named.n_postponed = task_get_n_tasks(system->postponed_task->rays);

      auto emission_time = std::chrono::duration_cast<std::chrono::microseconds>(emission_ptime - start_ptime);
      auto ray_caster_time = std::chrono::duration_cast<std::chrono::microseconds>(ray_caster_ptime - emission_ptime);
      auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(processing_ptime - ray_caster_ptime);

      system->params.stats->times.named.t_emission += emission_time.count();
      system->params.stats->times.named.t_ray_casting += ray_caster_time.count();
      system->params.stats->times.named.t_processing += processing_time.count();

      ++system->params.stats->n_steps;
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
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
