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

#include <map>
#include <thrust/system/cuda/vector.h>

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

    int n_mtgp_blocks;
    int support_double_atomic;

    struct auto_block_size_t
    {
      int n_make_ray_decisions;
      int n_filter_interaction;
      int n_ray_absorption;
      int n_ray_absorption_d;
      int n_ray_diffusion;
      int n_ray_reflection;
      int n_ray_transmition;
      int n_ray_postponement;
    };

    auto_block_size_t block_size;

    context_t host_context;
    context_t* device_context;

    mtgp32_kernel_params *kernel_params;

    subject::scene_t* scene;

    emission::task_t* emission_task;

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
  __host__ __device__ inline int align_interactions_count(int n)
  {
    const int n_interaction_alignment = 32;
    return n + (n_interaction_alignment - n % n_interaction_alignment) % n_interaction_alignment;
  }

  void prepare_context(system_t* system, int n_rays)
  {
    context_t& host_context = system->host_context;

    bool sync = false;
    if (host_context.n_rays_alloc < n_rays)
    {
      host_context.n_rays_alloc = align_rays_count(n_rays);
      if (host_context.hit_decision)
        cudaFree(host_context.hit_decision);
      checkCudaErrors(cudaMalloc(&host_context.hit_decision, sizeof(char) * host_context.n_rays_alloc));
      sync = true;
    }

    if (sync)
      sync_context_to_device(system);
  }

  __global__ void setup_ray_random(curandStatePhilox4_32_10_t *state)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
  }

  void contex_free_scene(context_t* ctx)
  {
    if (ctx->mesh_power)
      cudaFree(ctx->mesh_power);
    ctx->mesh_power = 0;
    if (ctx->mesh_power_d)
      cudaFree(ctx->mesh_power_d);
    ctx->mesh_power_d = 0;
  }

  int shutdown(system_t* system)
  {
    system->scene = 0;

    contex_free_scene(&system->host_context);

    cudaFree(system->host_context.HSGenR);
    cudaFree(system->host_context.HSGenTheta);
    cudaFree(system->host_context.PhiloxGenDecision);
    cudaFree(system->host_context.hit_decision);

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
    checkCudaErrors(cudaMalloc((void**)&host_context.mesh_power_d, sizeof(double) * scene->n_meshes));

    sync_context_to_device(system);

    return THERMAL_EQUATION_OK;
  }

  enum hit_decision
  {
    hit_diffuse_front = 0,
    hit_diffuse_rear = 1,
    hit_reflect = 2,
    hit_transmit = 3,
    hit_absorb = 4, // must be last interaction

    hit_decision_count = 5,
    hit_interaction_count = 5
  };

  __device__ inline int lane_id(void)
  {
    return threadIdx.x % warpSize;
  }

  __device__ inline int warp_broadcast(int value, int leader)
  {
    return __shfl(value, leader);
  }

  __global__ void make_ray_decisions(int n_rays, int* hit_face_idx, const ray_t* rays, curandStatePhilox4_32_10_t* ray_random,
    const float3* face_normal,
    const int* face_to_mesh,
    const subject::material_t* materials,
    const int* mesh_material,
    char* hit_decision,
    int* interaction_counts
  )
  {
    __shared__ int totals[hit_decision_count];
    if (threadIdx.x == 0)
    {
      for (int i = 0; i != hit_decision_count; ++i)
        totals[i] = 0;
    }
    __syncthreads();

    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandStatePhilox4_32_10_t local_state = ray_random[ray_idx];

    int counts[hit_decision_count] = {};

    for (; ray_idx < n_rays; ray_idx += stride)
    {
      const int face_idx = hit_face_idx[ray_idx];
      ray_t ray = rays[ray_idx];
      float3 normal = face_normal[face_idx];
      const int mesh_idx = face_to_mesh[face_idx];
      float hit_dice = curand_uniform(&local_state);
      bool hit_front = cuda_math::ray_check_hit_front(ray, normal);

      const subject::material_t& material = materials[mesh_material[mesh_idx]];
      const subject::optical_properties_t& optical = hit_front ? material.front.optical : material.rear.optical;
      
      char decision;
      if (hit_dice < optical.absorbance)
      {
        hit_face_idx[ray_idx] = mesh_idx;
        decision = hit_absorb;
      }
      else if (hit_dice < optical.absorbance + optical.specular_reflectance)
        decision = hit_reflect;
      else if (hit_dice < optical.absorbance + optical.specular_reflectance + optical.diffuse_reflectance)
      {
        if (hit_front)
          decision = hit_diffuse_front;
        else
          decision = hit_diffuse_rear;
      }
      else
        decision = hit_transmit;

      ++counts[decision];
      hit_decision[ray_idx] = decision;
    }

    ray_random[blockIdx.x * blockDim.x + threadIdx.x] = local_state;

    __syncthreads();
    for (int d = 0; d != hit_decision_count; ++d)
    {
      int value = counts[d];
      for (int i = 1; i <= 16; i *= 2)
      {
        int n = __shfl_up(value, i);
        if (lane_id() >= i)
          value += n;
      }

      if (lane_id() == warpSize - 1)
        atomicAdd(&totals[d], value);
    }

    __syncthreads();
    if (threadIdx.x == 0)
      for (int d = 0; d != hit_decision_count; ++d)
        atomicAdd(&interaction_counts[d], totals[d]);
  }

  __global__ void interaction_counts_to_offsets(int* data)
  {
    int prev = 0;
    for (int i = 0; i != hit_interaction_count; ++i)
    {
      int next = align_interactions_count(data[i]);
      data[i] = prev;
      prev += next;
    }
  }

  __global__ void filter_interaction(int n_rays, const char* hit_decision, int* interactions_offsets,
    const ray_t* s_ray, const float* s_hit_distance, const float* s_power, const int* s_hit_face_idx,
    ray_t* t_ray, float* t_hit_distance, float* t_power, int* t_hit_face_idx
  )
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; id < n_rays; id += stride)
    {
      const char decision = hit_decision[id];
      ray_t ray_buffer = s_ray[id];
      float hit_distance_buffer = s_hit_distance[id];
      float power_buffer = s_power[id];
      int hit_face_buffer = s_hit_face_idx[id];

      int active;
      for (int i = 0; i != hit_interaction_count; ++i)
        if (i == decision)
          active = __ballot(1);

      int leader = __ffs(active) - 1;
      int warp_offset;
      if (lane_id() == leader)
      {
        int warp_progress = __popc(active);
        warp_offset = atomicAdd(&interactions_offsets[decision], warp_progress);
      }
      warp_offset = warp_broadcast(warp_offset, leader);

      int thread_offset = warp_offset + __popc(active & ((1 << lane_id()) - 1));

      t_ray[thread_offset] = ray_buffer;
      t_hit_distance[thread_offset] = hit_distance_buffer;
      t_power[thread_offset] = power_buffer;
      t_hit_face_idx[thread_offset] = hit_face_buffer;
    }
  }

  __global__ void ray_absorption(int n_rays, const int* hit_mesh_idx, const float* ray_power, float* mesh_power)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      int mesh_idx = hit_mesh_idx[ray_idx];
      atomicAdd(&mesh_power[mesh_idx], ray_power[ray_idx]);
    }
  }

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
  __device__ double atomicAdd(double* a, double b) { return b; }
#endif

  __global__ void ray_absorption_d(int n_rays, const int* hit_mesh_idx, const float* ray_power, double* mesh_power)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      int mesh_idx = hit_mesh_idx[ray_idx];
      atomicAdd(&mesh_power[mesh_idx], ray_power[ray_idx]);
    }
  }

  template <bool front>
  __global__ void ray_diffusion(context_t* ctx, int n_rays, const int* hit_face_idx, const float* hit_distance, const float3* face_normals, ray_t* rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      int face_idx = hit_face_idx[ray_idx];
      float3 hit_point = ray_point(rays[ray_idx], hit_distance[ray_idx]);
      float3 face_normal = face_normals[face_idx];
      float3 malley_ray = malley_cuda::pick_uniform_ray(ctx);

      if (!front)
        malley_ray = cuda_math::invert_z(malley_ray);

      rays[ray_idx] = malley_cuda::ray_malley_emmit(malley_ray, face_normal, hit_point);
    }
  }

  __global__ void ray_reflection(int n_rays, const int* hit_face_idx, const float* hit_distance, const float3* face_normals, ray_t* rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      ray_t ray = rays[ray_idx];
      int face_idx = hit_face_idx[ray_idx];
      float3 hit_point = cuda_math::ray_point(ray, hit_distance[ray_idx]);
      float3 face_normal = face_normals[face_idx];

      rays[ray_idx] = cuda_math::ray_reflect(ray, face_normal, hit_point);
    }
  }

  __global__ void ray_transmition(int n_rays, const float* hit_distance, ray_t* rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      ray_t ray = rays[ray_idx];
      float3 hit_point = cuda_math::ray_point(ray, hit_distance[ray_idx]);

      rays[ray_idx] = cuda_math::ray_transmit(ray, hit_point);
    }
  }

  __global__ void ray_postponement(int n_rays, const ray_t* rays, const float* powers, ray_t* postponed_rays, float* postponed_powers)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      postponed_rays[ray_idx] = rays[ray_idx];
      postponed_powers[ray_idx] = powers[ray_idx];
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

    // Level-up emission tasks
    system_relocate_task(system->params.ray_caster, system->emission_task->rays, 0);
    for (int i = 0; i != params->n_emitters; ++i)
    {
      system_relocate_task(params->emitters[i], system->emission_task, 0);
    }

    system->face_to_mesh_index = 0;

    int n_mtgp_blocks = 128;
    system->n_mtgp_blocks = n_mtgp_blocks;

    int cuda_device = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, cuda_device);

    int _not_used;
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_make_ray_decisions, make_ray_decisions, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_filter_interaction, filter_interaction, 0, 0);
    system->block_size.n_filter_interaction = 256;
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_absorption, ray_absorption, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_absorption_d, ray_absorption_d, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_diffusion, ray_diffusion<true>, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_reflection, ray_reflection, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_transmition, ray_transmition, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&_not_used, &system->block_size.n_ray_postponement, ray_postponement, 0, 0);

    context_t& host_context = system->host_context;
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenR, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenTheta, n_mtgp_blocks * sizeof(curandStateMtgp32)));

    if (device_prop.major >= 6)
    {
      system->support_double_atomic = true;
      fprintf(stderr, "Using double-precision for mesh absorption.\n");
    } else
    {
      system->support_double_atomic = false;
      fprintf(stderr, "Using single-precision for mesh absorption.\n");
    }

    checkCudaErrors(cudaMalloc((void**)&system->kernel_params, sizeof(mtgp32_kernel_params)));
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, system->kernel_params));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenR, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 3));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenTheta, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 4));

    checkCudaErrors(cudaMalloc((void **)&host_context.PhiloxGenDecision, system->n_mtgp_blocks * system->block_size.n_make_ray_decisions * sizeof(curandStatePhilox4_32_10_t)));
    setup_ray_random << <system->n_mtgp_blocks, system->block_size.n_make_ray_decisions >> >(host_context.PhiloxGenDecision);

    checkCudaErrors(cudaMalloc((void **)&host_context.interaction_counts, sizeof(int) * hit_decision_count));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return THERMAL_EQUATION_OK;
  }

  union counts_layout_t
  {
    struct named_t
    {
      int n_diffused_front;
      int n_diffused_rear;
      int n_reflected;
      int n_transmitted;
      int n_absorbed;
      int n_missed;
    } named;

    static counts_layout_t interactions()
    {
      return counts_layout_t();
    }

    static counts_layout_t filtered_offsets(const counts_layout_t& c)
    {
      counts_layout_t l;
      l.n[0] = 0;
      for (int i = 1; i != hit_interaction_count; ++i)
        l.n[i] = l.n[i - 1] + align_interactions_count(c.n[i - 1]);
      return l;
    }

    static counts_layout_t compact_offsets(const counts_layout_t& c)
    {
      counts_layout_t l;
      int prev = 0;
      for (int i = 0; i != hit_decision_count; ++i)
      {
        int next = c.n[i];
        l.n[i] = prev;
        prev += next;
      }
      return l;
    }

    int postponed() const
    {
      return named.n_diffused_front + named.n_diffused_rear + named.n_reflected + named.n_transmitted;
    }

    int n[hit_decision_count];
  };

  int calculate(system_t* system, thermal_equation::task_t* task)
  {
    const int n_meshes = system->scene->n_meshes;

    int r = 0;

    // Prepare emission tasks
    emission::task_t* emission_task = system->emission_task;
    task_next_gen(emission_task->rays);

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

    task_reserve(&emission_task->rays, n_interaction_alignment * hit_interaction_count);

    ray_caster::task_t* ray_caster_task = emission_task->rays;
    const int n_rays = ray_caster_task->n_tasks;
    const int n_missed = ray_caster_task->n_missed;

    if ((r = scene_relocate(system->scene, 1)) < 0)
      return r;

    prepare_context(system, n_rays);
    cudaMemset(system->host_context.interaction_counts, 0, sizeof(int) * hit_decision_count);

    subject::scene_t* scene = system->scene;
    make_ray_decisions << <system->n_mtgp_blocks, system->block_size.n_make_ray_decisions >> >(n_rays,
      ray_caster_task->hit_face_idx,
      (ray_t*)ray_caster_task->ray,
      system->host_context.PhiloxGenDecision,
      (const float3*)scene->face_normals,
      scene->face_to_mesh,
      scene->materials,
      scene->mesh_material,
      system->host_context.hit_decision,
      system->host_context.interaction_counts);

    counts_layout_t n_counts;
    cudaMemcpy(n_counts.n, system->host_context.interaction_counts, sizeof(int) * hit_decision_count, cudaMemcpyDeviceToHost);
    interaction_counts_to_offsets << <1, 1 >> > (system->host_context.interaction_counts);

    const int n_postponed = n_counts.postponed();

    counts_layout_t n_filtered_offsets = counts_layout_t::filtered_offsets(n_counts);
    counts_layout_t n_compact_offsets = counts_layout_t::compact_offsets(n_counts);

    const int n_filter_interaction_grid = (n_rays + system->block_size.n_filter_interaction - 1) / system->block_size.n_filter_interaction;
    const ray_caster::rays_data_t* source = ray_caster_task;
    ray_caster::rays_data_t* filtered = &ray_caster_task->next_gen;
    filter_interaction << <n_filter_interaction_grid, system->block_size.n_filter_interaction >> > (n_rays,
      system->host_context.hit_decision,
      system->host_context.interaction_counts, // interaction offsets now
      (ray_t*)source->ray,
      source->hit_distance,
      source->power,
      source->hit_face_idx,
      (ray_t*)filtered->ray,
      filtered->hit_distance,
      filtered->power,
      filtered->hit_face_idx
      );

    // @detail Yet alignment it is not crucial for ray index resolving.
    // @todo Does align of interaction groups on cache-line boundary speedup further interaction kernel application?
    // @todo Without alignment we can run single postpone kernel.

    if (n_counts.named.n_absorbed)
    {
      if (system->support_double_atomic)
      {
        cudaMemset(system->host_context.mesh_power_d, 0, sizeof(double) * n_meshes);
        const int n_ray_absorption_grid = (n_counts.named.n_absorbed + system->block_size.n_ray_absorption_d - 1) /
                                          system->block_size.n_ray_absorption_d;
        ray_absorption_d << < n_ray_absorption_grid, system->block_size.n_ray_absorption_d >> > (
                n_counts.named.n_absorbed,
                        filtered->hit_face_idx +
                        n_filtered_offsets.named.n_absorbed, // hit to mesh idx for absorption rays now
                        filtered->power + n_filtered_offsets.named.n_absorbed,
                        system->host_context.mesh_power_d
        );
      }
      else
      {
        cudaMemset(system->host_context.mesh_power, 0, sizeof(float) * n_meshes);
        const int n_ray_absorption_grid = (n_counts.named.n_absorbed + system->block_size.n_ray_absorption - 1) /
                                          system->block_size.n_ray_absorption;
        ray_absorption << < n_ray_absorption_grid, system->block_size.n_ray_absorption >> > (
                n_counts.named.n_absorbed,
                        filtered->hit_face_idx +
                        n_filtered_offsets.named.n_absorbed, // hit to mesh idx for absorption rays now
                        filtered->power + n_filtered_offsets.named.n_absorbed,
                        system->host_context.mesh_power
        );
      }
    }

    if (n_counts.named.n_diffused_front)
    {
      // Grid and block size is limited by random generator.
      ray_diffusion<true> << <system->n_mtgp_blocks, 256 >> > (
        system->device_context,
        n_counts.named.n_diffused_front,
        filtered->hit_face_idx + n_filtered_offsets.named.n_diffused_front,
        filtered->hit_distance + n_filtered_offsets.named.n_diffused_front,
        (const float3*)scene->face_normals,
        (ray_t*)filtered->ray + n_filtered_offsets.named.n_diffused_front
        );
    }

    if (n_counts.named.n_diffused_rear)
    {
      // Grid and block size is limited by random generator.
      ray_diffusion<false> << <system->n_mtgp_blocks, 256 >> > (
        system->device_context,
        n_counts.named.n_diffused_rear,
        filtered->hit_face_idx + n_filtered_offsets.named.n_diffused_rear,
        filtered->hit_distance + n_filtered_offsets.named.n_diffused_rear,
        (const float3*)scene->face_normals,
        (ray_t*)filtered->ray + n_filtered_offsets.named.n_diffused_rear
        );
    }

    if (n_counts.named.n_reflected)
    {
      const int n_ray_reflected_grid = (n_counts.named.n_reflected + system->block_size.n_ray_reflection - 1) / system->block_size.n_ray_reflection;
      ray_reflection << <n_ray_reflected_grid, system->block_size.n_ray_reflection >> > (
        n_counts.named.n_reflected,
        filtered->hit_face_idx + n_filtered_offsets.named.n_reflected,
        filtered->hit_distance + n_filtered_offsets.named.n_reflected,
        (const float3*)scene->face_normals,
        (ray_t*)filtered->ray + n_filtered_offsets.named.n_reflected
        );
    }

    if (n_counts.named.n_transmitted)
    {
      const int n_ray_transmitted_grid = (n_counts.named.n_transmitted + system->block_size.n_ray_transmition - 1) / system->block_size.n_ray_transmition;
      ray_transmition << <n_ray_transmitted_grid, system->block_size.n_ray_transmition >> > (
        n_counts.named.n_transmitted,
        filtered->hit_distance + n_filtered_offsets.named.n_transmitted,
        (ray_t*)filtered->ray + n_filtered_offsets.named.n_transmitted
        );
    }

    int n_fill_source = hit_absorb - 1;
    int n_fill_source_size = n_counts.n[n_fill_source];
    for (int i = 0; i < n_fill_source; ++i)
    {
      int n_hole_size = n_filtered_offsets.n[i] - n_compact_offsets.n[i];
      while (n_counts.n[i] && n_hole_size && i != n_fill_source)
      {
        int n_can_fill = std::min(n_fill_source_size, n_hole_size);
        if (!n_can_fill)
        {
          --n_fill_source;
          n_fill_source_size = n_counts.n[n_fill_source];
          continue;
        }

        int n_source_offset = n_compact_offsets.n[n_fill_source] + n_fill_source_size - n_can_fill;

        cudaMemcpy(filtered->ray + n_compact_offsets.n[i], filtered->ray + n_source_offset, sizeof(ray_t) * n_can_fill, cudaMemcpyDeviceToDevice);
        cudaMemcpy(filtered->power + n_compact_offsets.n[i], filtered->power + n_source_offset, sizeof(float) * n_can_fill, cudaMemcpyDeviceToDevice);

        n_fill_source_size -= n_can_fill;
        n_hole_size -= n_can_fill;
      }
    }

    filtered->n_tasks = n_postponed;

    if (system->support_double_atomic)
    {
      double *h_mesh_power = (double *) malloc(n_meshes * sizeof(double));
      checkCudaErrors(cudaMemcpy(h_mesh_power, system->host_context.mesh_power_d, n_meshes * sizeof(double),
                                 cudaMemcpyDeviceToHost));
      for (int i = 0; i != n_meshes; ++i)
        task->absorption[i] += h_mesh_power[i];
      free(h_mesh_power);

    }
    else
    {
      float *h_mesh_power = (float *) malloc(n_meshes * sizeof(float));
      checkCudaErrors(cudaMemcpy(h_mesh_power, system->host_context.mesh_power, n_meshes * sizeof(float),
                                 cudaMemcpyDeviceToHost));
      for (int i = 0; i != n_meshes; ++i)
        task->absorption[i] += h_mesh_power[i];
      free(h_mesh_power);
    }

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
