// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
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

#include "optix_hit_converter.h"
#include "optix_hit_converter.cuh"
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "../cuda_misc/cuda_math.cuh"
#include "../cuda_misc/cuda_check_errors.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../cuda_misc/cuda_math.cuh"

namespace optix_hit_converter
{
  __device__ float3 triangle_uv_to_point(face_t face, float u, float v)
  {
    return u * face.points[0] + v * face.points[1] + (1.f - u - v) * face.points[2];
  }
#if CUDART_VERSION > 6
  __device__ inline int lane_id(void)
  {
    return threadIdx.x % warpSize;
  }

  __device__ inline int warp_broadcast(int value, int leader)
  {
    return __shfl_sync(0xffffffff, value, leader);
  }

  __global__ void convert_hits(const optix_hit_t* hits, int n_rays, int* offset,
    const cuda_math::ray_t* s_ray, const float* s_power,
    cuda_math::ray_t* t_ray, int* t_hit_face_idx, float* t_hit_distance, float* t_power
    )
  {
    int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {  
      const optix_hit_t hit = hits[ray_idx];

      if (hit.distance > 0)
      {
        int active = __ballot_sync(0xffffffff, 1);
        int leader = __ffs(active) - 1;
        int warp_offset;
        if (lane_id() == leader)
        {
          int warp_progress = __popc(active);
          warp_offset = atomicAdd(offset, warp_progress);
        }
        warp_offset = warp_broadcast(warp_offset, leader);

        int thread_offset = warp_offset + __popc(active & ((1 << lane_id()) - 1));

        t_ray[thread_offset] = s_ray[ray_idx];
        t_power[thread_offset] = s_power[ray_idx];
        t_hit_face_idx[thread_offset] = hit.triangle_id;
        t_hit_distance[thread_offset] = hit.distance;
      }
    }
  }

#else
    __global__ void convert_hits(const optix_hit_t* hits, int n_rays, int* offset,
                                 const cuda_math::ray_t* s_ray, const float* s_power,
                                 cuda_math::ray_t* t_ray, int* t_hit_face_idx, float* t_hit_distance, float* t_power
    )
    {
      int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
      int stride = gridDim.x * blockDim.x;
      for (; ray_idx < n_rays; ray_idx += stride)
      {
        const optix_hit_t hit = hits[ray_idx];

        if (hit.distance > 0)
        {
          int thread_offset = atomicAdd(offset, 1);

          t_ray[thread_offset] = s_ray[ray_idx];
          t_power[thread_offset] = s_power[ray_idx];
          t_hit_face_idx[thread_offset] = hit.triangle_id;
          t_hit_distance[thread_offset] = hit.distance;
        }
      }
    }

#endif

  bool resolve_supported()
  {
    int count = 0;
    return cudaSuccess == cudaGetDeviceCount(&count) && count > 0;
  }  

  bool is_supported()
  {
    static const bool result = resolve_supported();
    return result;
  }

  bool is_supported(const state_t* state)
  {
    return state->optix_buffer_type == 0x201;
  }

  state_t* state_init(subject::scene_t* scene, bool force_cpu)
  {
    state_t* state = (state_t*)calloc(1, sizeof(state_t));

    bool supported = is_supported() && !force_cpu;

    state->optix_buffer_type = supported ? 0x201 : 0x200; // RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST

    state->scene = scene;
    if (supported)
    {
      int minGridSize;
      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &state->convert_block_size, convert_hits, 0, 0);

      cudaMalloc(&state->n_offset, sizeof(int));
    }

    return state;
  }

  void state_free(state_t* state)
  {
    if (!state)
      return;
    
    if (is_supported(state))
    {
      cudaFree(state->hits);
      cudaFree(state->n_offset);
    }
    else
      free(state->hits);

    free(state);
  }

  int state_prepare(state_t* state, ray_caster::task_t* task)
  {
    if (state->n_hits >= task->n_tasks)
      return RAY_CASTER_OK;

    bool cuda_enabled = is_supported(state);
    const bool allocation_valid = (cuda_enabled && task->allocation_device >= 0) ||
      (!cuda_enabled && task->allocation_device < 0);

    if (!allocation_valid)
      return -RAY_CASTER_INVALID_TASK_ALLOCATION;

    state->n_hits = (task->n_tasks + 2047) & -2048;

    if (cuda_enabled)
    {
      if (state->hits)
        cudaFree(state->hits);
      checkCudaErrors(cudaMalloc((void**)&state->hits, sizeof(optix_hit_t) * task->n_tasks));
    }
    else
    {
      state->hits = (optix_hit_t*)realloc(state->hits, state->n_hits * sizeof(optix_hit_t));
    }

    return RAY_CASTER_OK;
  }

  int convert_cuda(state_t* state, ray_caster::task_t* task)
  {
    if (int r = scene_relocate(state->scene, 0))
    {
      if (r == -SCENE_ENGINE_NOT_SUPPORTED)
        fprintf(stderr, "Optix ray caster requires CUDA enabled scene!\n");
      return r;
    }

    cudaMemset(state->n_offset, 0, sizeof(int));

    ray_caster::rays_data_t* source = task;
    ray_caster::rays_data_t* target = &task->next_gen;
    int n_grid_size = (task->n_tasks + state->convert_block_size - 1) / state->convert_block_size;
    convert_hits << <n_grid_size, state->convert_block_size >> > (state->hits,
      task->n_tasks , state->n_offset,
      (const cuda_math::ray_t*)source->ray, source->power,
      (cuda_math::ray_t*)target->ray, target->hit_face_idx, target->hit_distance, target->power);

    int n_filtered;
    int n_tasks_before = task->n_tasks;
    cudaMemcpy(&n_filtered, state->n_offset, sizeof(int), cudaMemcpyDeviceToHost);
    task_next_gen(task);
    task->n_missed = n_tasks_before - n_filtered;
    task->n_tasks = n_filtered;

    // @todo Remove it
    // @note This is scene, not task!
    if (int r = scene_relocate(state->scene, -1))
      return r;

    return RAY_CASTER_OK;
  }

  int convert_cpu(state_t* state, ray_caster::task_t* task)
  {
    if (int r = scene_relocate(state->scene, -1))
      return r;

    int n_tasks_before = task->n_tasks;
    ray_caster::rays_data_t* source = task;
    ray_caster::rays_data_t* target = &task->next_gen;
    for (int r = 0; r != task->n_tasks; ++r)
    {
      const optix_hit_t& hit = state->hits[r];
      if (hit.distance > 0)
      {
        int& n_offset = target->n_tasks;
        target->ray[n_offset] = source->ray[r];
        target->hit_face_idx[n_offset] = hit.triangle_id;
        target->hit_distance[n_offset] = hit.distance;
        target->power[n_offset] = source->power[r];
        ++n_offset;
      }
    }

    task_next_gen(task);
    task->n_missed = n_tasks_before - task->n_tasks;

    return RAY_CASTER_OK;
  }  

  int convert(state_t* state, ray_caster::task_t* task)
  {
    return is_supported(state) ? convert_cuda(state, task) : convert_cpu(state, task);
  }
}
