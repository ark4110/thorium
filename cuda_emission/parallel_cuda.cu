// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

#include "cuda_emission.h"
#include "parallel_cuda.cuh"

#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include "../math/ray.h"
#include <float.h>
#include <random>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

#include <helper_cuda.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <iostream>

namespace {
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
}

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

namespace cuda_parallel_rays_emission
{
  using namespace cuda_math;

  struct system_t : emission::system_t
  {
    params_t params;

    context_t host_context;
    context_t* device_context;

    int n_tpb;
    int n_warp_size;
    int n_mtgp_blocks;

    mtgp32_kernel_params *kernel_params;
  };

  void sync_context_to_device(system_t* system)
  {
    if (system->device_context == 0)
      checkCudaErrors(cudaMalloc((void**)&system->device_context, sizeof(context_t)));
    checkCudaErrors(cudaMemcpy(system->device_context, &system->host_context, sizeof(context_t), cudaMemcpyHostToDevice));
  }

  void align_rays_count_to_warp_size(params_t& params, int n_warp_size)
  {
    for (int source_idx = 0; source_idx != params.n_sources; ++source_idx)
    {
      parallel_rays_emission_cpu::parallel_rays_source_t& source = params.sources[source_idx];
      int align = (n_warp_size - source.n_rays % n_warp_size) % n_warp_size;
      source.n_rays += align;
    }
  }

  int calculate_total_rays(const params_t& params)
  {
    int n_rays = 0;
    for (int i = 0; i != params.n_sources; ++i)
      n_rays += params.sources[i].n_rays;
    return n_rays;
  }

  /// @brief Initializes system with given ray caster after creation.
  int init(system_t* system, const parallel_rays_emission_cpu::params_t* params)
  {
    system->params = *params;

    system->host_context = {};
    system->device_context = 0;
    system->kernel_params = 0;

    const char* argv[] = { "" };
    int device_id = findCudaDevice(1, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_id));
    system->n_tpb = deviceProp.maxThreadsPerBlock;
    system->n_warp_size = deviceProp.warpSize;

    align_rays_count_to_warp_size(system->params, system->n_warp_size);

    int n_mtgp_blocks = 64;
    system->n_mtgp_blocks = n_mtgp_blocks;

    context_t& host_context = system->host_context;
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenR, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenTheta, n_mtgp_blocks * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void**)&system->kernel_params, sizeof(mtgp32_kernel_params)));
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, system->kernel_params));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenR, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 3));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenTheta, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 4));

    return EMISSION_OK;
  }

  int shutdown(system_t* system)
  {
    cudaFree(system->host_context.HSGenR);
    cudaFree(system->host_context.HSGenTheta);

    system->host_context = {};
    cudaFree(system->device_context);
    system->device_context = 0;

    cudaFree(system->kernel_params);
    system->kernel_params = 0;

    return EMISSION_OK;
  }

  void update_host_context(system_t* system, subject::scene_t* scene)
  {
    math::sphere_t bsphere = math::triangles_bsphere(scene->faces, scene->n_faces);
    float radius = bsphere.radius * parallel_rays_emission_cpu::scene_bsphere_emission_scale;

    struct local_context_t
    {
      math::vec3 origin;
      float radius;
    };

    local_context_t local = {};
    local.origin = bsphere.center;
    local.radius = radius;
    memcpy(&system->host_context, &local, sizeof(local_context_t));
  }

  int set_scene(system_t* system, subject::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    update_host_context(system, scene);
    sync_context_to_device(system);

    return EMISSION_OK;
  }

  __global__ void emit_parallel_rays(context_t* ctx, source_params_t params, ray_t* rays, float* powers)
  {
    float3 origin = ctx->origin;
    float radius = ctx->radius;

    float offset = 1.3f + radius;
    const mat33 rotation = rotate_towards(make_float3(0, 0, 1), params.direction);
    float3 direction = params.direction;
    int n_rays = params.n_rays;

    const float ray_power = params.power * (float(M_PI) * ctx->radius * ctx->radius) / (float)n_rays;

    curandStateMtgp32* HSGenR = &ctx->HSGenR[blockIdx.x];
    curandStateMtgp32* HSGenTheta = &ctx->HSGenTheta[blockIdx.x];

    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += blockDim.x * gridDim.x)
    {
      float r = radius * sqrtf(curand_uniform(HSGenR));
      float theta = M_2PI * curand_uniform(HSGenTheta);

      float s, c;
      sincosf(theta, &s, &c);

      float3 ray_origin = rotation * make_float3(r * c, r * s, -offset) + origin;
      rays[ray_idx] = { ray_origin, direction };
      powers[ray_idx] = ray_power;
    }
  }

  __global__ void emit_multiple_parallel_rays(context_t* ctx, const multi_source_params_t* params, int n_sources, ray_t* rays, float* powers)
  {
    float3 origin = ctx->origin;
    float radius = ctx->radius;

    int source_idx = blockIdx.x;
    for (; source_idx < n_sources; source_idx += gridDim.x)
    {
      const multi_source_params_t param = params[source_idx];

      float offset = 1.3f + radius;
      const float3 direction = param.direction;
      const mat33 rotation = rotate_towards(make_float3(0, 0, 1), direction);
      int n_rays_offset = param.n_rays_offset;
      int n_rays = param.n_rays;

      const float ray_power = param.power * (float(M_PI) * ctx->radius * ctx->radius) / (float)n_rays;

      curandStateMtgp32* HSGenR = &ctx->HSGenR[blockIdx.x];
      curandStateMtgp32* HSGenTheta = &ctx->HSGenTheta[blockIdx.x];

      int ray_idx = threadIdx.x;
      for (; ray_idx < n_rays; ray_idx += blockDim.x)
      {
        float r = radius * sqrtf(curand_uniform(HSGenR));
        float theta = M_2PI * curand_uniform(HSGenTheta);

        float s, c;
        sincosf(theta, &s, &c);

        float3 ray_origin = rotation * make_float3(r * c, r * s, -offset) + origin;
        rays[ray_idx + n_rays_offset] = { ray_origin, direction };
        powers[ray_idx + n_rays_offset] = ray_power;
      }
    }
  }

  int calculate_flat(system_t* system, emission::task_t* task)
  {
    align_rays_count_to_warp_size(system->params, system->n_warp_size);
    int n_total_rays = calculate_total_rays(system->params);
    const int n_input_rays = task_grow(task, n_total_rays);

    ray_t* device_rays = (ray_t*)task->rays->ray + n_input_rays;
    float* device_powers = task->rays->power + n_input_rays;
    const int n_sources = system->params.n_sources;

    // Launch kernels
    for (int source_idx = 0; source_idx != n_sources; ++source_idx)
    {
      parallel_rays_emission_cpu::parallel_rays_source_t& source = system->params.sources[source_idx];
      if (source.n_rays == 0)
        continue;

      source_params_t params = { make_float3(source.direction.x, source.direction.y, source.direction.z), source.power, source.n_rays };
      emit_parallel_rays << <system->n_mtgp_blocks, 256 >> > (system->device_context, params, device_rays, device_powers);
      device_rays += source.n_rays;
      device_powers += source.n_rays;
    }

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return EMISSION_OK;
  }

  int calculate_grid(system_t* system, emission::task_t* task)
  {
    align_rays_count_to_warp_size(system->params, system->n_warp_size);
    int n_total_rays = calculate_total_rays(system->params);
    const int n_input_rays = task_grow(task, n_total_rays);
    
    ray_t* device_rays = (ray_t*)task->rays->ray + n_input_rays;
    float* device_powers = task->rays->power + n_input_rays;
    
    const int n_sources = system->params.n_sources;

    multi_source_params_t* host_multi_source_params = (multi_source_params_t*)calloc(n_sources, sizeof(multi_source_params_t));
    int offset = 0;
    int n_grid_threshold = n_total_rays * 3 / n_sources;
    int n_grid_count = 0;
    for (int source_idx = 0; source_idx != n_sources; ++source_idx)
    {
      const parallel_rays_emission_cpu::parallel_rays_source_t& f = system->params.sources[source_idx];
      multi_source_params_t& m = host_multi_source_params[source_idx];
      memcpy(&m.direction, &f.direction, sizeof(float3));
      m.n_rays = f.n_rays < n_grid_threshold ? f.n_rays : 0;
      if (f.n_rays < n_grid_threshold)
        ++n_grid_count;
      m.n_rays_offset = offset;
      m.power = f.power;
      offset += f.n_rays;
    }

    multi_source_params_t* device_multi_source_params = 0;
    cudaMalloc(&device_multi_source_params, sizeof(multi_source_params_t) * n_sources);
    cudaMemcpy(device_multi_source_params, host_multi_source_params, sizeof(multi_source_params_t) * n_sources, cudaMemcpyHostToDevice);

    emit_multiple_parallel_rays << <system->n_mtgp_blocks, 128 >> > (system->device_context, device_multi_source_params, n_sources, device_rays, device_powers);

    for (int source_idx = 0; source_idx != n_sources; ++source_idx)
    {
      const parallel_rays_emission_cpu::parallel_rays_source_t& f = system->params.sources[source_idx];
      if (f.n_rays >= n_grid_threshold)
      {
        source_params_t params = { make_float3(f.direction.x, f.direction.y, f.direction.z), f.power, f.n_rays };
        int n_rays_offset = host_multi_source_params[source_idx].n_rays_offset;
        emit_parallel_rays << <system->n_mtgp_blocks, 256 >> > (system->device_context, params, device_rays + n_rays_offset, device_powers + n_rays_offset);
      }
    }

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(device_multi_source_params);

    free(host_multi_source_params);

    return EMISSION_OK;
  }

  int calculate(system_t* system, emission::task_t* task)
  {
    if (task->rays && task->rays->allocation_device <= 0)
      return -EMISSION_INVALID_TASK_ALLOCATION;

    //return calculate_flat(system, task);
    if (system->params.n_sources < 5)
      return calculate_flat(system, task);
    else
      return calculate_grid(system, task);
  }

  /// @brief Creates virtual methods table from local methods.
  const emission::system_methods_t methods =
  {
    (int(*)(emission::system_t* system, const emission::params_t*))&init,
    (int(*)(emission::system_t* system))&shutdown,
    (int(*)(emission::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(emission::system_t* system, emission::task_t* task))&calculate,
    &cuda_emission::cuda_task_relocation
  };

  emission::system_t* system_create()
  {
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
