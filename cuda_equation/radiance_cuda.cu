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
#include "../thermal_equation/radiance_cpu.h"
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

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

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
    int n_postpone_blocks;

    struct auto_block_size_t
    {
      int n_make_ray_random;
      int n_ray_absorption;
      int n_ray_diffusion;
      int n_ray_reflection;
      int n_ray_transmition;
      int n_ray_postponement;
    };

    auto_block_size_t block_size;

    cudaStream_t* streams;

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

  const int n_interaction_alignment = 32;
  const int n_interaction_types_count = 5;

  int align_interactions_count(int n)
  {
    return n + (n_interaction_alignment - n % n_interaction_alignment) % n_interaction_alignment;
  }

  void prepare_context(system_t* system, int n_rays)
  {
    context_t& host_context = system->host_context;

    bool sync = false;
    if (host_context.n_rays_alloc < n_rays)
    {
      host_context.n_rays_alloc = align_rays_count(n_rays);
      if (host_context.ray_random)
        cudaFree(host_context.ray_random);
      if (host_context.hit_decision)
        cudaFree(host_context.hit_decision);
      if (host_context.ray_interaction_idx)
        cudaFree(host_context.ray_interaction_idx);
      checkCudaErrors(cudaMalloc(&host_context.ray_random, sizeof(float) * host_context.n_rays_alloc));
      checkCudaErrors(cudaMalloc(&host_context.hit_decision, sizeof(char) * host_context.n_rays_alloc));
      checkCudaErrors(cudaMalloc(&host_context.ray_interaction_idx, sizeof(int) * (host_context.n_rays_alloc + 4 * n_interaction_alignment)));
      sync = true;
    }

    if (sync)
      sync_context_to_device(system);
  }

  __global__ void setup_ray_random(curandStatePhilox4_32_10_t *state)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1, id, 0, &state[id]);
  }

  __global__ void make_ray_random(context_t* ctx, int n_rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t local_state = ctx->PhiloxGenDecision[ray_idx];
    float* ray_random = ctx->ray_random;
    for (; ray_idx < n_rays; ray_idx += stride)
      ray_random[ray_idx] = curand_uniform(&local_state);
    ctx->PhiloxGenDecision[blockIdx.x * blockDim.x + threadIdx.x] = local_state;
  }

  void contex_free_scene(context_t* ctx)
  {
    if (ctx->mesh_power)
      cudaFree(ctx->mesh_power);
    ctx->mesh_power = 0;
  }

  int shutdown(system_t* system)
  {
    system->scene = 0;

    contex_free_scene(&system->host_context);

    cudaFree(system->host_context.HSGenR);
    cudaFree(system->host_context.HSGenTheta);
    cudaFree(system->host_context.PhiloxGenDecision);
    cudaFree(system->host_context.ray_random);
    cudaFree(system->host_context.hit_decision);
    cudaFree(system->host_context.postponed_offsets);
    cudaFree(system->host_context.ray_interaction_idx);

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
    subject::scene_t* ray_caster_scene = (subject::scene_t*)scene; // slicing binary compatible scenes

    int r = 0;
    if ((r = system_set_scene(system->params.ray_caster, ray_caster_scene)) < 0)
      return r;

    if ((r = system_update(system->params.ray_caster)) < 0)
      return r;

    for (int e = 0; e != system->params.n_emitters; ++e)
      if ((r = system_set_scene(system->params.emitters[e], scene)) < 0)
        return r;

    context_t& host_context = system->host_context;

    checkCudaErrors(cudaMalloc((void**)&host_context.mesh_power, sizeof(float) * scene->n_meshes)); 

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
    hit_missed = 5,
    hit_diffuse_front = 6,
    hit_diffuse_rear = 7,
  };

  __device__ hit_decision make_hit_decision(float hit_dice, const subject::optical_properties_t& optical)
  {
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

  __global__ void make_ray_decisions(int n_rays, int* hit_face_idx, const ray_t* rays, const float* ray_random,
                                      const float3* face_normal,
                                      const int* face_to_mesh,
                                      const subject::material_t* materials,
                                      const int* mesh_material,
                                      char* hit_decision,
                                      int* reducer_count
  )
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int absorbed = 0;
    int diffused_front = 0;
    int diffused_rear = 0;
    int reflected = 0;
    int transmitted = 0;
    int missed = 0;

    for (; ray_idx < n_rays; ray_idx += stride)
    {
      const int face_idx = hit_face_idx[ray_idx];
      if (face_idx < 0)
      {
        ++missed;
        hit_decision[ray_idx] = hit_missed;
        continue;
      }

      ray_t ray = rays[ray_idx];
      float3 normal = face_normal[face_idx];
      bool hit_front = cuda_math::ray_check_hit_front(ray, normal);

      const int mesh_idx = face_to_mesh[face_idx];
      const subject::material_t& material = materials[mesh_material[mesh_idx]];
      const subject::optical_properties_t& optical = hit_front ? material.front.optical : material.rear.optical;

      float hit_dice = ray_random[ray_idx];
      char decision;
      if (hit_dice < optical.absorbance)
      {
        ++absorbed;
        decision = hit_absorb;
      }
      else if (hit_dice < optical.absorbance + optical.specular_reflectance)
      {
        ++reflected;
        decision = hit_reflect;
      }
      else if (hit_dice < optical.absorbance + optical.specular_reflectance + optical.diffuse_reflectance)
      {
        if (hit_front)
        {
          ++diffused_front;
          decision = hit_diffuse_front;
        }
        else
        {
          ++diffused_rear;
          decision = hit_diffuse_rear;
        }
      }
      else if (hit_dice < optical.absorbance + optical.specular_reflectance + optical.diffuse_reflectance + optical.transmittance)
      {
        ++transmitted;
        decision = hit_transmit;
      }
      else
        decision = hit_drop;
      hit_decision[ray_idx] = decision;
    }

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    reducer_count[thread_id + 0 * stride] = absorbed;
    reducer_count[thread_id + 1 * stride] = diffused_front;
    reducer_count[thread_id + 2 * stride] = diffused_rear;
    reducer_count[thread_id + 3 * stride] = reflected;
    reducer_count[thread_id + 4 * stride] = transmitted;
  }

  __global__ void resolve_interaction(int n_rays, const char* hit_decision, const int* thread_offset, int n_diffused_front_offset, int n_diffused_rear_offset, int n_reflected_offset, int n_transmitted_offset, int* ray_interaction_idx)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int* absorbed = ray_interaction_idx + thread_offset[id];
    int* diffused_front = ray_interaction_idx + n_diffused_front_offset + thread_offset[id + 1 * stride];
    int* diffused_rear = ray_interaction_idx + n_diffused_rear_offset + thread_offset[id + 2 * stride];
    int* reflected = ray_interaction_idx + n_reflected_offset + thread_offset[id + 3 * stride];
    int* transmitted = ray_interaction_idx + n_transmitted_offset + thread_offset[id + 4 * stride];

    for (; id < n_rays; id += stride)
    {
      switch (hit_decision[id])
      {
        case hit_diffuse_front:
          (*diffused_front++) = id;
          break;

        case hit_diffuse_rear:
          (*diffused_rear++) = id;
          break;

        case hit_absorb:
          (*absorbed++) = id;
          break;

        case hit_reflect:
          (*reflected++) = id;
          break;

        case hit_transmit:
          (*transmitted++) = id;

        default:
          break;
      }
    }
  }

  __global__ void ray_absorption(int n_rays, const int* interaction_ray_idx, const int* hit_face_idx, const int* face_to_mesh, const float* ray_power, float* mesh_power)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; id < n_rays; id += stride)
    {
      int ray_idx = interaction_ray_idx[id];
      int face_idx = hit_face_idx[ray_idx];
      int mesh_idx = face_to_mesh[face_idx];
      atomicAdd(&mesh_power[mesh_idx], ray_power[ray_idx]);
    }
  }

  template <bool front>
  __global__ void ray_diffusion(context_t* ctx, int n_rays, const int* interaction_ray_idx, const int* hit_face_idx, const float3* hit_points, const float3* face_normals, ray_t* rays)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; id < n_rays; id += stride)
    {
      int ray_idx = interaction_ray_idx[id];
      int face_idx = hit_face_idx[ray_idx];
      float3 hit_point = hit_points[ray_idx];
      float3 face_normal = face_normals[face_idx];
      float3 malley_ray = malley_cuda::pick_uniform_ray(ctx);
      
      if (!front)
        malley_ray = cuda_math::invert_z(malley_ray);

      rays[ray_idx] = malley_cuda::ray_malley_emmit(malley_ray, face_normal, hit_point);
    }
  }

  __global__ void ray_reflection(int n_rays, const int* interaction_ray_idx, const int* hit_face_idx, const float3* hit_points, const float3* face_normals, ray_t* rays)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; id < n_rays; id += stride)
    {
      int ray_idx = interaction_ray_idx[id];
      const ray_t& ray = rays[ray_idx];
      int face_idx = hit_face_idx[ray_idx];
      float3 hit_point = hit_points[ray_idx];
      float3 face_normal = face_normals[face_idx];

      rays[ray_idx] = cuda_math::ray_reflect(ray, face_normal, hit_point);
    }
  }

  __global__ void ray_transmition(int n_rays, const int* interaction_ray_idx, const float3* hit_points, ray_t* rays)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; id < n_rays; id += stride)
    {
      int ray_idx = interaction_ray_idx[id];
      const ray_t& ray = rays[ray_idx];
      float3 hit_point = hit_points[ray_idx];
      
      rays[ray_idx] = cuda_math::ray_transmit(ray, hit_point);
    }
  }

  __global__ void ray_postponement(int n_rays, const int* interaction_ray_idx, const ray_t* rays, const float* powers, ray_t* postponed_rays, float* postponed_powers)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; id < n_rays; id += stride)
    {
      int ray_idx = interaction_ray_idx[id];
      postponed_rays[id] = rays[ray_idx];
      postponed_powers[id] = powers[ray_idx];
    }
  }

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

    int n_mtgp_blocks = 128;
    system->n_mtgp_blocks = n_mtgp_blocks;

    int _not_used;
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_make_ray_random, make_ray_random, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_absorption, ray_absorption, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_diffusion, ray_diffusion<true>, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_reflection, ray_reflection, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_transmition, ray_transmition, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_postponement, ray_postponement, 0, 0);

    context_t& host_context = system->host_context;
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenR, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenTheta, n_mtgp_blocks * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void**)&system->kernel_params, sizeof(mtgp32_kernel_params)));
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, system->kernel_params));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenR, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 3));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenTheta, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 4));

    checkCudaErrors(cudaMalloc((void **)&host_context.PhiloxGenDecision, n_mtgp_blocks * system->block_size.n_make_ray_random * sizeof(curandStatePhilox4_32_10_t)));
    setup_ray_random << <system->n_mtgp_blocks, system->block_size.n_make_ray_random >> >(host_context.PhiloxGenDecision);

    system->n_postpone_blocks = 256;
    
    checkCudaErrors(cudaMalloc((void **)&host_context.postponed_offsets, sizeof(int) * n_interaction_types_count * system->n_postpone_blocks * system->n_tpb));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return THERMAL_EQUATION_OK;
  }

  union interaction_counts_t
  {
    struct named_t
    {
      int n_absorbed;
      int n_diffused_front;
      int n_diffused_rear;
      int n_reflected;
      int n_transmitted;
    } named;

    int postponed() const
    {
      return named.n_diffused_front + named.n_diffused_rear + named.n_reflected + named.n_transmitted;
    }

    int n[n_interaction_types_count];
  };

  union source_interaction_offsets_t
  {
    source_interaction_offsets_t(const interaction_counts_t& c)
    {
      n[0] = 0;
      for (int i = 1; i != n_interaction_types_count; ++i)
        n[i] = n[i - 1] + align_interactions_count(c.n[i - 1]);
    }

    struct named_t
    {
      int n_absorbed;
      int n_diffused_front;
      int n_diffused_rear;
      int n_reflected;
      int n_transmitted;
    } named;
    int n[n_interaction_types_count];
  };

  union target_interaction_offsets_t
  {
    target_interaction_offsets_t(const interaction_counts_t& c)
    {
      n[0] = 0;
      for (int i = 1; i != n_interaction_types_count; ++i)
        n[i] = n[i - 1] + c.n[i];
    }

    struct named_t
    {
      int n_absorbed;
      int n_diffused_front;
      int n_diffused_rear;
      int n_reflected;
      int n_transmitted;
    } named;
    int n[n_interaction_types_count];
  };

  int calculate(system_t* system, thermal_equation::task_t* task)
  {
    const int n_meshes = system->scene->n_meshes;

    int r = 0;

    // Prepare emission tasks
    std::swap(system->emission_task, system->postponed_task);
    emission::task_t* emission_task = system->emission_task;
    emission::task_t* postponed_task = system->postponed_task;
    emission_task->mesh_emission = task->emission;
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

    task_grow(system->postponed_task, n_rays);
    if ((r = system_relocate_task(system->params.ray_caster, postponed_task->rays, 1)) < 0)
      return r;

    if ((r = scene_relocate(system->scene, 1)) < 0)
      return r;

    prepare_context(system, n_rays);
    cudaMemset(system->host_context.mesh_power, 0, sizeof(float) * n_meshes);

    make_ray_random << <system->n_mtgp_blocks, system->block_size.n_make_ray_random>> > (system->device_context, n_rays);

    subject::scene_t* scene = system->scene;
    make_ray_decisions<<<system->n_postpone_blocks, system->n_tpb>>>(n_rays,
                        ray_caster_task->hit_face_idx,
                        (ray_t*)ray_caster_task->ray,
                        system->host_context.ray_random,
                        (const float3*)scene->face_normals,
                        scene->face_to_mesh,
                        scene->materials,
                        scene->mesh_material,
                        system->host_context.hit_decision,
                        system->host_context.postponed_offsets);
    const int n_reducer_stride = system->n_postpone_blocks * system->n_tpb;
    thrust::device_ptr<int> postponed_offsets(system->host_context.postponed_offsets);

    interaction_counts_t n_counts;
    
    for (int i = 0; i != n_interaction_types_count; ++i)
      n_counts.n[i] = thrust::reduce(postponed_offsets + i * n_reducer_stride, postponed_offsets + (i + 1) * n_reducer_stride);
    
    int n_postponed = n_counts.postponed();
    const int n_missed = n_rays - n_counts.named.n_absorbed - n_postponed;
    for (int i = 0; i != n_interaction_types_count; ++i)
      thrust::exclusive_scan(postponed_offsets + i * n_reducer_stride, postponed_offsets + (i + 1) * n_reducer_stride, postponed_offsets + i * n_reducer_stride);

    // @detail Yet alignment it is not crucial for ray index resolving.
    // @todo Does align of interaction groups on cache-line boundary speedup further interaction kernel application?
    // @todo Without alignment we can run single postpone kernel.

    source_interaction_offsets_t n_source_offsets(n_counts);
    target_interaction_offsets_t n_target_offsets(n_counts);

    resolve_interaction<<<system->n_postpone_blocks, system->n_tpb>>>(n_rays,
            system->host_context.hit_decision,
            system->host_context.postponed_offsets,
            n_source_offsets.named.n_diffused_front,
            n_source_offsets.named.n_diffused_rear,
            n_source_offsets.named.n_reflected,
            n_source_offsets.named.n_transmitted,
            system->host_context.ray_interaction_idx);

    // Now we know values of rays indices and rays count for each interaction type. So we can just run interaction kernels on known ranges.
    
    int* ray_interaction_idx = system->host_context.ray_interaction_idx;

    if (n_counts.named.n_absorbed)
    {
      const int n_ray_absorption_grid = (n_counts.named.n_absorbed + system->block_size.n_ray_absorption - 1) / system->block_size.n_ray_absorption;
      ray_absorption << <n_ray_absorption_grid, system->block_size.n_ray_absorption >> > (
        n_counts.named.n_absorbed,
        ray_interaction_idx,
        ray_caster_task->hit_face_idx,
        scene->face_to_mesh,
        ray_caster_task->power,
        system->host_context.mesh_power
        );
    }

    if (n_counts.named.n_diffused_front)
    {
      // Grid and block size is limited by random generator.
      ray_diffusion<true> << <system->n_mtgp_blocks, 256 >> > (
        system->device_context,
        n_counts.named.n_diffused_front,
        ray_interaction_idx + n_source_offsets.named.n_diffused_front,
        ray_caster_task->hit_face_idx,
        (const float3*)ray_caster_task->hit_point,
        (const float3*)scene->face_normals,
        (ray_t*)ray_caster_task->ray
        );
    }

    if (n_counts.named.n_diffused_rear)
    {
      // Grid and block size is limited by random generator.
      ray_diffusion<false> << <system->n_mtgp_blocks, 256 >> > (
        system->device_context,
        n_counts.named.n_diffused_rear,
        ray_interaction_idx + n_source_offsets.named.n_diffused_rear,
        ray_caster_task->hit_face_idx,
        (const float3*)ray_caster_task->hit_point,
        (const float3*)scene->face_normals,
        (ray_t*)ray_caster_task->ray
        );
    }

    if (n_counts.named.n_reflected)
    {
      const int n_ray_reflected_grid = (n_counts.named.n_reflected + system->block_size.n_ray_reflection - 1) / system->block_size.n_ray_reflection;
      ray_reflection << <n_ray_reflected_grid, system->block_size.n_ray_reflection >> > (
        n_counts.named.n_reflected,
        ray_interaction_idx + n_source_offsets.named.n_reflected,
        ray_caster_task->hit_face_idx,
        (const float3*)ray_caster_task->hit_point,
        (const float3*)scene->face_normals,
        (ray_t*)ray_caster_task->ray
        );
    }

    if (n_counts.named.n_transmitted)
    {
      const int n_ray_transmitted_grid = (n_counts.named.n_transmitted + system->block_size.n_ray_transmition - 1) / system->block_size.n_ray_transmition;
      ray_transmition << <n_ray_transmitted_grid, system->block_size.n_ray_transmition >> > (
        n_counts.named.n_transmitted,
        ray_interaction_idx + n_source_offsets.named.n_transmitted,
        (const float3*)ray_caster_task->hit_point,
        (ray_t*)ray_caster_task->ray
        );
    }
    
    for (int i = 1; i != n_interaction_types_count; ++i)
    {
      // @note There are holes because of alignment.
      // @todo Don't align and perform single kernel run?
      int n_postponed = n_counts.n[i];
      if (n_postponed)
      {
        int n_source_offset = n_source_offsets.n[i];
        int n_target_offset = n_target_offsets.n[i - 1];
        const int n_ray_postponed_grid = (n_postponed + system->block_size.n_ray_postponement - 1) / system->block_size.n_ray_postponement;
        ray_postponement << <n_ray_postponed_grid, system->block_size.n_ray_postponement >> > (
          n_postponed,
          ray_interaction_idx + n_source_offset,
          (const ray_t*)ray_caster_task->ray,
          ray_caster_task->power,
          (ray_t*)postponed_task->rays->ray + n_target_offset,
          postponed_task->rays->power + n_target_offset
          );
      }
    }

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    postponed_task->rays->n_tasks = n_postponed;

    float* h_mesh_power = (float*)malloc(n_meshes * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_mesh_power, system->host_context.mesh_power, n_meshes * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i != n_meshes; ++i)
      task->absorption[i] += h_mesh_power[i];    
    free(h_mesh_power);

    auto processing_ptime = std::chrono::high_resolution_clock().now();

    if (system->params.stats)
    {
      auto& ray_stats = system->params.stats->quants.named;
      ray_stats.n_postponed = n_postponed;
      ray_stats.n_absorbed = n_counts.named.n_absorbed;
      ray_stats.n_diffused = n_counts.named.n_diffused_front + n_counts.named.n_diffused_rear;
      ray_stats.n_reflected = n_counts.named.n_reflected;
      ray_stats.n_transmited = n_counts.named.n_transmitted;
      ray_stats.n_missed = n_missed;

      auto emission_time = std::chrono::duration_cast<std::chrono::microseconds>(emission_ptime - start_ptime);
      auto ray_caster_time = std::chrono::duration_cast<std::chrono::microseconds>(ray_caster_ptime - emission_ptime);
      auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(processing_ptime - ray_caster_ptime);

      system->params.stats->times.named.t_emission += (int)emission_time.count();
      system->params.stats->times.named.t_ray_casting += (int)ray_caster_time.count();
      system->params.stats->times.named.t_processing += (int)processing_time.count();

      ++system->params.stats->n_steps;
    }

    // @todo Don't relocate back, every scene client must handle relocation itself.
    if ((r = scene_relocate(system->scene, -1)) < 0)
      return r;

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
