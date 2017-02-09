// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
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

#include "optix_hit_converter.h"
#include "optix_hit_converter.cuh"
#include <stdlib.h>
#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include "../math/operations.h"
#include "../math/triangle.h"

namespace optix_hit_converter
{
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

  state_t* state_init(const ray_caster::scene_t* scene)
  {
    state_t* state = (state_t*)calloc(1, sizeof(state_t));

    bool supported = is_supported();

    state->optix_buffer_type = supported ? 0x201 : 0x200; // RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST

    state->host_faces = scene->faces;
    if (supported)
    {
      checkCudaErrors(cudaMalloc((void**)&state->device_faces, sizeof(ray_caster::face_t) * scene->n_faces));
      checkCudaErrors(cudaMemcpy(state->device_faces, scene->faces, sizeof(ray_caster::face_t) * scene->n_faces, cudaMemcpyHostToDevice));
    }   

    return state;
  }

  void state_free(state_t* state)
  {
    if (state)
    {
      cudaFree(state->device_faces);

      if (is_supported())
      {
        cudaFree(state->hits);
        cudaFree(state->device_hit_face_idx);
        cudaFree(state->device_hit_points);
      }
      else
        free(state->hits);
    }
    free(state);
  }

  void state_prepare(state_t* state, ray_caster::task_t* task)
  {
    if (state->n_hits >= task->n_tasks)
      return;

    state->n_hits = task->n_tasks;

    if (optix_hit_converter::is_supported())
    {
      if (state->hits)
        cudaFree(state->hits);
      checkCudaErrors(cudaMalloc((void**)&state->hits, sizeof(optix_hit_t) * task->n_tasks));

      if (state->device_hit_points)
        cudaFree(state->device_hit_points);      
      checkCudaErrors(cudaMalloc(&state->device_hit_points, sizeof(float3) * task->n_tasks));

      if (state->device_hit_face_idx)
        cudaFree(state->device_hit_face_idx);
      checkCudaErrors(cudaMalloc(&state->device_hit_face_idx, sizeof(int) * task->n_tasks));
    }
    else
    {
      state->hits = (optix_hit_t*)realloc(state->hits, task->n_tasks * sizeof(optix_hit_t));
    }
  }

  __device__ float3 triangle_uv_to_point(face_t face, float u, float v)
  {
    return u * face.points[0] + v * face.points[1] + (1.f - u - v) * face.points[2];
  }

  __global__ void convert_hits(const optix_hit_t* __restrict__ hits, const face_t* __restrict__ faces, int n_rays, float3* hit_point, int* hit_face_idx)
  {
    int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += gridDim.x * blockDim.x)
    {
      const optix_hit_t& hit = hits[ray_idx];
      if (hit.distance < 0)
        hit_face_idx[ray_idx] = -1;
      else
      {
        hit_face_idx[ray_idx] = hit.triangle_id;
        hit_point[ray_idx] = triangle_uv_to_point(faces[hit.triangle_id], hit.u, hit.v);
      }
    }    
  }

  int convert_cuda(state_t* state, ray_caster::task_t* task)
  {
    const int n_rays = task->n_tasks;

    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
    convert_hits << <128, 128 >> > (state->hits, (const face_t*)state->device_faces, task->n_tasks, (float3*)state->device_hit_points, state->device_hit_face_idx);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaThreadSetCacheConfig(cudaFuncCachePreferNone);

    cudaMemcpy(task->hit_point, state->device_hit_points, sizeof(float3) * n_rays, cudaMemcpyDeviceToHost);
    cudaMemcpy(task->hit_face_idx, state->device_hit_face_idx, sizeof(int) * n_rays, cudaMemcpyDeviceToHost);

    return RAY_CASTER_OK;
  }

  int convert_cpu(state_t* state, ray_caster::task_t* task)
  {
    for (int r = 0; r != task->n_tasks; ++r)
    {
      const optix_hit_t& hit = state->hits[r];
      if (hit.distance < 0)
        task->hit_face_idx[r] = -1;
      else
      {
        task->hit_face_idx[r] = hit.triangle_id;
        task->hit_point[r] = math::triangle_uv_to_point(state->host_faces[hit.triangle_id], hit.u, hit.v);
      }
    }

    return RAY_CASTER_OK;
  }  

  int convert(state_t* state, ray_caster::task_t* task)
  {
    return is_supported() ? convert_cuda(state, task) : convert_cpu(state, task);
  }
}
