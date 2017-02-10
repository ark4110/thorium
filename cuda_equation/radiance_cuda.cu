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
    system->host_context.n_meshes = system->scene->n_meshes;

    if (system->device_context == 0)
      checkCudaErrors(cudaMalloc((void**)&system->device_context, sizeof(context_t)));
    checkCudaErrors(cudaMemcpy(system->device_context, &system->host_context, sizeof(context_t), cudaMemcpyHostToDevice));
  }

  int align_rays_count(int n_rays)
  {
    const int alignment = 1024;
    return n_rays + (alignment - n_rays % alignment) % alignment;
  }

  void prepare_context(system_t* system, int n_rays)
  {
    context_t& host_context = system->host_context;

    if (host_context.n_rays_alloc < n_rays)
    {
      host_context.n_rays_alloc = align_rays_count(n_rays);
      if (host_context.hit_descision)
        cudaFree(host_context.hit_descision);
      checkCudaErrors(cudaMalloc(&host_context.hit_descision, sizeof(char) * host_context.n_rays_alloc));

      if (host_context.ray_power)
        cudaFree(host_context.ray_power);
      checkCudaErrors(cudaMalloc(&host_context.ray_power, sizeof(float) * host_context.n_rays_alloc));

      if (host_context.postponed_ray_power)
        cudaFree(host_context.postponed_ray_power);
      checkCudaErrors(cudaMalloc(&host_context.postponed_ray_power, sizeof(float) * host_context.n_rays_alloc));

      sync_context_to_device(system);
    }
  }

  int get_postponed_rays_count(system_t* system)
  {
    context_t from_device;
    checkCudaErrors(cudaMemcpy(&from_device, system->device_context, sizeof(context_t), cudaMemcpyDeviceToHost));
    return from_device.n_postponed;
  }

  /// @brief Initializes system with given ray caster after creation.
  int init(system_t* system, params_t* params)
  {
    assert(sizeof(float3) == sizeof(math::vec3));
    assert(sizeof(ray_t) == sizeof(math::ray_t));

    system->params = *params;

    system->host_context = {};
    system->device_context = 0;
    system->kernel_params = 0;

    system->emission_task = emission::task_create();    
    system->postponed_task = emission::task_create();

    // Level-up emission tasks
    system_relocate_task(system->params.ray_caster, system->emission_task->rays, 0);
    system_relocate_task(system->params.ray_caster, system->postponed_task->rays, 0);
    for (int i = 0; i != params->n_emitters; ++i)
    {
      system_relocate_task(params->emitters[i], system->emission_task, 0);
      system_relocate_task(params->emitters[i], system->postponed_task, 0);
    }

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

    if (ctx->mesh_powers)
      cudaFree(ctx->mesh_powers);
    ctx->mesh_powers = 0;

    if (ctx->ray_power)
      cudaFree(ctx->ray_power);
    ctx->ray_power = 0;

    if (ctx->postponed_ray_power)
      cudaFree(ctx->postponed_ray_power);
    ctx->postponed_ray_power = 0;
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

    checkCudaErrors(cudaMalloc((void**)&host_context.mesh_powers, sizeof(float) * scene->n_meshes)); 

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

  __global__ void make_ray_descisions(context_t* ctx, int n_rays, const int* hit_face_idx, const float3* hit_point, ray_t* rays)
  {
    char* hit_descisions = ctx->hit_descision;
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += blockDim.x * gridDim.x)
    {
      const int face_idx = hit_face_idx[ray_idx];
      if (face_idx < 0)
      {
        hit_descisions[ray_idx] = hit_missed;
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

      hit_descisions[ray_idx] = decision;
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

  inline __device__ bool is_postpone_descision(char descision)
  {
    return descision == hit_reflect || descision == hit_diffuse || descision == hit_transmit;
  }

  /**
    Single block of n_tpb threads.
    Shared n_thread_postponed size is n_tpb * sizeof(int).
  */
  __global__ void postpone_rays(context_t* ctx, int n_rays, const ray_t* rays, ray_t* postponed_ray)
  {
    extern __shared__ int s_thread_postponed[];

    // Init shared memory.
    s_thread_postponed[threadIdx.x] = 0;

    const char* hit_descision = ctx->hit_descision;
    

    // Calculate postponed rays count for each thread.
    int r = threadIdx.x;
    for (; r < n_rays; r += blockDim.x)
    {
      char descision = hit_descision[r];
      if (is_postpone_descision(descision))
        s_thread_postponed[threadIdx.x] += 1;
    }
    __syncthreads();
    
     //Gather step for leading thread: calculate each thread offset
     //and report total postponed count.
     //@todo Parallelize?
    if (threadIdx.x == 0)
    { 
      int acc = s_thread_postponed[0];
      for (int i = 1; i < blockDim.x; ++i)
      {
        // @todo Check s_thread_postponed[i] = s_thread_postponed[i] + s_thread_postponed[i - 1]
        int new_acc = acc + s_thread_postponed[i];
        s_thread_postponed[i] = acc;
        acc = new_acc;
      }
      ctx->n_postponed = acc;
    }
    __syncthreads();

    const float* raw_powers = ctx->ray_power;
    float* postponed_raw_powers = ctx->postponed_ray_power;

    r = threadIdx.x;
    int postponed_index = s_thread_postponed[threadIdx.x];
    for (; r < n_rays; r += blockDim.x)
    {      
      char descision = hit_descision[r];
      if (is_postpone_descision(descision))
      {
        if (postponed_index >= ctx->n_rays_alloc)
          printf("postponed %d\n", postponed_index);
        postponed_ray[postponed_index] = rays[r];
        postponed_raw_powers[postponed_index] = raw_powers[r];
        ++postponed_index;
      }
    }
  }

  /**
    Each block process mesh_power_shared_size meshes.
    Each thread process all rays.
    I.e. suboptional implementation for mesh count > 12k
  */
  __global__ void calculate_mesh_absorption(const context_t* ctx, int n_rays, const int* hit_face_idx)
  {
    extern __shared__ float s_mesh_power[];

    const int mesh_power_shared_size = 48 * 1024 / sizeof(float);

    // Init shared memory.
    const int n_meshes = ctx->n_meshes;
    for (int tid = threadIdx.x; tid < n_meshes; tid += blockDim.x)
      s_mesh_power[tid] = 0;
    __syncthreads();

    const int* face_to_mesh_idx = ctx->face_to_mesh_index;
    const char* hit_descision = ctx->hit_descision;
    const float* raw_powers = ctx->ray_power;
    int ray_idx = threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += blockDim.x)
    {
      const char descision = hit_descision[ray_idx];
      if (descision == hit_absorb)
      {
        const int mesh_idx = face_to_mesh_idx[hit_face_idx[ray_idx]];
        if (mesh_idx / mesh_power_shared_size == blockIdx.x)
        {
          atomicAdd(s_mesh_power + (mesh_idx % mesh_power_shared_size), raw_powers[ray_idx]);
        }
      }
    }
    __syncthreads();

    // Assign accumulated mesh powers.
    float* mesh_power = ctx->mesh_powers;
    const int block_mesh_offset = mesh_power_shared_size * blockIdx.x;
    for (int tid = threadIdx.x; tid < mesh_power_shared_size && block_mesh_offset + tid < n_meshes; tid += blockDim.x)
    {
      mesh_power[block_mesh_offset + tid] = s_mesh_power[tid];
    }
  }

  int calculate(system_t* system, thermal_equation::task_t* task)
  {
    int r = 0;

    // Prepare emission tasks
    std::swap(system->emission_task, system->postponed_task);
    emission::task_t* emission_task = system->emission_task;
    emission::task_t* postponed_task = system->postponed_task;
    task_reset(postponed_task);

    if (system->params.stats)
      *system->params.stats = stats_t();

    emission_task->mesh_temperatures = task->temperatures;
    emission_task->mesh_emission = task->emission;

    auto start_ptime = std::chrono::high_resolution_clock().now();

    for (int e = 0; e != system->params.n_emitters; ++e)
    { 
      emission::system_t* emitter = system->params.emitters[e];
      // Right now emission expect host task
      if ((r = system_relocate_task(emitter, emission_task, 0)) < 0)
        return r;
      if ((r = system_calculate(emitter, system->emission_task)) < 0)
        return r;
    }
    auto emission_ptime = std::chrono::high_resolution_clock().now();

    // @todo Is nothing to do ok?
    if (system->emission_task->rays == 0)
      return THERMAL_EQUATION_OK;

    // Put task to ray caster native device.
    if ((r = system_relocate_task(system->params.ray_caster, emission_task->rays, 0)) < 0)
      return r;
    if ((r = system_cast(system->params.ray_caster, emission_task->rays)) < 0)
      return r;
    // Put ray caster result to current CUDA device.
    if ((r = system_relocate_task(system->params.ray_caster, emission_task->rays, 1 /* @todo use current device*/)) < 0)
      return r;
    auto ray_caster_ptime = std::chrono::high_resolution_clock().now();

    const ray_caster::task_t* ray_caster_task = emission_task->rays;
    const int n_rays = ray_caster_task->n_tasks;

    prepare_context(system, n_rays);
    
    make_ray_descisions << <system->n_mtgp_blocks, 256 >> > (system->device_context
      , n_rays
      , ray_caster_task->hit_face_idx
      , (float3*)ray_caster_task->hit_point
      , (ray_t*)ray_caster_task->ray);

    ////// Postpone step    
    // Emission powers are on host device.
    checkCudaErrors(cudaMemcpy(system->host_context.ray_power, emission_task->power, sizeof(float) * n_rays, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    task_grow(system->postponed_task, n_rays);
    if ((r = system_relocate_task(system->params.ray_caster, postponed_task->rays, 1)) < 0)
      return r;

    int shared_size = system->n_tpb * sizeof(int);
    postpone_rays << <1, system->n_tpb, shared_size >> > (
      system->device_context
      , n_rays
      , (ray_t*)ray_caster_task->ray // mutated rays
      , (ray_t*)postponed_task->rays->ray);
    
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    const int n_postponed = get_postponed_rays_count(system);
    checkCudaErrors(cudaMemcpy(system->postponed_task->power, system->host_context.postponed_ray_power, sizeof(float) * n_postponed, cudaMemcpyDeviceToHost));
    postponed_task->rays->n_tasks = n_postponed;

    ////// Gather step
    const int mesh_power_shared_size = 48 * 1024;
    const int n_mesh_power_shared_size = mesh_power_shared_size / sizeof(float);
    const int n_mesh_blocks = (system->scene->n_meshes + n_mesh_power_shared_size - 1) / n_mesh_power_shared_size;
    calculate_mesh_absorption<<<n_mesh_blocks, system->n_tpb, mesh_power_shared_size>>> (system->device_context
      , n_rays
      , ray_caster_task->hit_face_idx);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    const int n_meshes = system->scene->n_meshes;
    float* mesh_powers = (float*)malloc(n_meshes * sizeof(float));
    checkCudaErrors(cudaMemcpy(mesh_powers, system->host_context.mesh_powers, n_meshes * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i != n_meshes; ++i)
      task->absorption[i] += mesh_powers[i];
    free(mesh_powers);

    /*for (int r = 0; r != n_rays; ++r)
    {
      const int hit_face_idx = ray_caster_task->hit_face_idx[r];
      float ray_power = system->postponed_task->power[r];

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

      switch (decision)
      {
      case hit_absorb:
        task->absorption[hit_mesh_idx] += ray_power;
        break;

      case hit_reflect:
      case hit_diffuse:
      case hit_transmit:
      break;

      case hit_drop:
      case hit_missed:
      default:
        break;
      }
    }*/
    
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
