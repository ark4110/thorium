// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

#include "cuda_emission.h"
#include "malley_cuda.cuh"
#include "../emission/malley_cpu.h"
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
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "../ext/cub/cub/cub.cuh"

namespace {
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
}

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

namespace cuda_malley_emission
{
  using namespace cuda_math;

  struct grid_calculator_t
  {
    template <typename  T>
    grid_calculator_t(T func)
    {
      cudaOccupancyMaxPotentialBlockSize(&n_min_blocks, &n_block_size, func, 0, 0);
    }

    int block(int n)
    {
      return std::min(std::max(n / n_min_blocks, 32), n_block_size);
    }

    int grid(int n)
    {
      int block_size = block(n);
      return (n + block_size - 1) / block_size;
    }

    int n_min_blocks;
    int n_block_size;
  };

  /// @brief Extended base system_t (C-style polymorphism)
  struct system_t : emission::system_t
  {
    params_t params;

    subject::scene_t* scene;

    curandStateMtgp32* GenA;
    curandStateMtgp32* GenB;
    curandStateMtgp32* GenR;
    curandStateMtgp32* GenTheta;
    mtgp32_kernel_params *kernel_params;

    float* mesh_temperature;
    float* face_front_weight;
    float* face_rear_weight;
    float* face_front_emission;
    float* face_rear_emission;
    int* face_front_rays_count;
    int* face_rear_rays_count;
    int* face_rays_offset;
    int* face_rays_count_next;

    void* reduce_buffer;
    size_t reduce_buffer_size;
    int* reduce_result_int;
    float* reduce_result_float;

    grid_calculator_t calculate_weights_grid;
    grid_calculator_t scale_emission_grid;
    grid_calculator_t calculate_rays_count_grid;

    cudaStream_t memcpy_stream;

    int n_mtgp_blocks;
  };



#define sigma 5.670400e-8f

  struct scene_ctx_t
  {
    int n_faces;
    const face_t* faces;
    const int* face_to_mesh;
    const float* face_area;
    const float3* face_normal;
    const int* mesh_material;
    const subject::material_t* materials;
  };

  scene_ctx_t make_scene_ctx(const subject::scene_t* scene)
  {
    scene_ctx_t r = {};
    r.n_faces = scene->n_faces;
    r.faces = (const face_t*)scene->faces;
    r.face_to_mesh = scene->face_to_mesh;
    r.face_area = scene->face_areas;
    r.face_normal = (const float3*)scene->face_normals;
    r.mesh_material = scene->mesh_material;
    r.materials = scene->materials;
    return r;
  }

  struct random_ctx_t
  {
    curandStateMtgp32* GenA;
    curandStateMtgp32* GenB;
    curandStateMtgp32* GenR;
    curandStateMtgp32* GenTheta;
  };

  random_ctx_t make_random_ctx(system_t* system)
  {
    random_ctx_t r = {};
    r.GenA = system->GenA;
    r.GenB = system->GenB;
    r.GenR = system->GenR;
    r.GenTheta = system->GenTheta;
    return r;
  }

  __global__ void calculate_weights(scene_ctx_t s,
    const float* temperatures,
    float* front_weight, float* rear_weight,
    float* front_emission, float* rear_emission
  )
  {
    int face_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; face_idx < s.n_faces; face_idx += stride)
    {
      int mesh_idx = s.face_to_mesh[face_idx];
      float area = s.face_area[face_idx];
      int mesh_material_idx = s.mesh_material[mesh_idx];
      const subject::material_t material = s.materials[mesh_material_idx];
      float front_potential = area * material.front.emission.emissivity;
      float rear_potential = area * material.rear.emission.emissivity;

      float T = temperatures[mesh_idx];
      float weight_density = T; // some empirical distribution function of temperature
      float power_density = sigma * (T * T * T * T);

      front_weight[face_idx] = weight_density * front_potential;
      rear_weight[face_idx] = weight_density * rear_potential;
      front_emission[face_idx] = power_density * front_potential;
      rear_emission[face_idx] = power_density * rear_potential;
    }
  }

  __global__ void calculate_rays_count(int n_faces, int n_rays, const float* weights, const float* front_weights, const float* rear_weights, int* front_counts, int* rear_counts)
  {
    int face_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float count_factor = (float)n_rays / (weights[0] + weights[1]);

    for (; face_idx < n_faces; face_idx += stride)
    {
      front_counts[face_idx] = (int)(ceilf(count_factor * front_weights[face_idx]));
      rear_counts[face_idx] = (int)(ceilf(count_factor * rear_weights[face_idx]));
    }
  }

  __global__ void scale_emission(int n_faces, const int* rays_count, float* emission)
  {
    int face_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; face_idx < n_faces; face_idx += stride)
      emission[face_idx] /= (float)rays_count[face_idx];
  }

  template <bool front_emission>
  __global__ void emit_malley_rays(scene_ctx_t s,
    int* rays_count,
    int* face_rays_offset,
    int global_offset,
    const float* emissions,
    ray_t* rays,
    float* power
  )
  {
    int face_idx = blockIdx.x;
    cuda_math::mat33 rotation = pick_face_rotation(s.face_normal[face_idx]);
    face_t face = s.faces[face_idx];
    float3 u = face.points[1] - face.points[0];
    float3 v = face.points[2] - face.points[0];
    float emission = emissions[face_idx];
    int n_face_rays = rays_count[face_idx];
    int offset = global_offset + face_rays_offset[face_idx];

    for (int r = threadIdx.x; r < n_face_rays; r += blockDim.x)
    {
      int ray_offset = r + offset;
      power[ray_offset] = emission;
      ray_t &ray = rays[ray_offset];
      float3 direction = rotation * (front_emission ? ray.direction : invert_z(
        ray.direction)); // we prepared random uniform direction in ray itself
      float a = ray.origin.x;
      float b = ray.origin.y;      
      float3 origin = face.points[0] + a * u + b * v;
      ray = { origin + direction * 0.0001f, direction };
    }
  }

  __global__ void generate_malley_random(random_ctx_t g, const int* n_rays_counts, ray_t* rays)
  {
    const int n_rays = n_rays_counts[0] + n_rays_counts[1];

    curandStateMtgp32* GenA = &g.GenA[blockIdx.x];
    curandStateMtgp32* GenB = &g.GenB[blockIdx.x];
    curandStateMtgp32* GenR = &g.GenR[blockIdx.x];
    curandStateMtgp32* GenTheta = &g.GenTheta[blockIdx.x];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; idx < n_rays; idx += stride)
    {
      float a = curand_uniform(GenA);
      float b = curand_uniform(GenB);
      if (a + b > 1)
      {
        a = 1.f - a;
        b = 1.f - b;
      }
      rays[idx].origin.x = a;
      rays[idx].origin.y = b;

      float r = curand_uniform(GenR);
      float theta = M_2PI * curand_uniform(GenTheta);

      float rad = sqrtf(r);
      float s, c;
      sincosf(theta, &s, &c);
      rays[idx].direction = make_float3(rad * c, rad * s, sqrtf(1 - r));
    }
  }

  void update_mesh_rays_emitted(system_t* system, emission::task_t* task)
  {
    scene_ctx_t s = make_scene_ctx(system->scene);
    const int n_faces = s.n_faces;
    const int n_meshes = system->scene->n_meshes;

    int* mesh_indices;
    int* ray_counts;
    cudaMalloc((void**)&mesh_indices, n_meshes * sizeof(int));
    cudaMalloc((void**)&ray_counts, n_meshes * sizeof(int));

    thrust::reduce_by_key(thrust::device,
      s.face_to_mesh,
      s.face_to_mesh + n_faces,
      system->face_front_rays_count,
      mesh_indices,
      ray_counts
    );
    cudaMemcpy(task->mesh_rays_emitted, ray_counts, n_meshes * sizeof(int), cudaMemcpyDeviceToHost);
    thrust::reduce_by_key(thrust::device,
      s.face_to_mesh,
      s.face_to_mesh + n_faces,
      system->face_rear_rays_count,
      mesh_indices,
      ray_counts
    );
    int* temp = (int*)malloc(n_meshes * sizeof(int));
    cudaMemcpy(temp, ray_counts, n_meshes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(mesh_indices);
    cudaFree(ray_counts);
    for (int m = 0; m != n_meshes; ++m)
      task->mesh_rays_emitted[m] += temp[m];
    free(temp);
  }

  __global__ void init_work_queue(int n_faces, int n_faces_per_block, int n_queue_size_per_block, int* queue, int* queue_head, int* queue_tail)
  {
    int idx = threadIdx.x;
    int stride = blockDim.x;

    int* block_queue = queue + blockIdx.x * n_queue_size_per_block;
    int* head = queue_head + 32 * blockIdx.x;
    int* tail = queue_tail + 32 * blockIdx.x;

    if (threadIdx.x == 0)
    {
      *head = 0;
      const int final_block = (n_faces + n_faces_per_block - 1) / n_faces_per_block;
      if (blockIdx.x + 1 < final_block)
        *tail = n_faces_per_block;
      else if (blockIdx.x + 1 == final_block)
        *tail = n_faces - (blockIdx.x * n_faces_per_block);
      else
        *tail = 0;
    }

    for (; idx < n_queue_size_per_block; idx += stride)
    {
      int face_idx = blockIdx.x * n_faces_per_block + idx;
      block_queue[idx] = face_idx < n_faces ? face_idx : -1;
    }
  }

  int calculate(system_t* system, emission::task_t* task)
  {
    if (task->rays && task->rays->allocation_device <= 0)
      return -EMISSION_INVALID_TASK_ALLOCATION;

    const int n_faces = system->scene->n_faces;
    const int n_meshes = system->scene->n_meshes;

    int r;
    if (r = scene_relocate(system->scene, 1))
      return r;

    scene_ctx_t scene_ctx = make_scene_ctx(system->scene);

    cudaMemcpy(system->mesh_temperature, task->mesh_temperatures, sizeof(float) * n_meshes, cudaMemcpyHostToDevice);

    // Calculate face weights
    auto g = system->calculate_weights_grid;
    calculate_weights << <g.grid(n_faces), g.block(n_faces) >> > (scene_ctx,
      system->mesh_temperature,
      system->face_front_weight, system->face_rear_weight,
      system->face_front_emission, system->face_rear_emission);

    cub::DeviceReduce::Sum(system->reduce_buffer, system->reduce_buffer_size, system->face_front_weight, &system->reduce_result_float[0], n_faces);
    cub::DeviceReduce::Sum(system->reduce_buffer, system->reduce_buffer_size, system->face_rear_weight, &system->reduce_result_float[1], n_faces);

    // Calculate rays count
    g = system->calculate_rays_count_grid;
    calculate_rays_count << <g.grid(n_faces), g.block(n_faces) >> > (n_faces, system->params.n_rays,
      system->reduce_result_float,
      system->face_front_weight, system->face_rear_weight,
      system->face_front_rays_count, system->face_rear_rays_count);

    if (task->mesh_rays_emitted)
      update_mesh_rays_emitted(system, task);

    // Calculate total rays count
    cub::DeviceReduce::Sum(system->reduce_buffer, system->reduce_buffer_size, system->face_front_rays_count, &system->reduce_result_int[0], n_faces);
    cub::DeviceReduce::Sum(system->reduce_buffer, system->reduce_buffer_size, system->face_rear_rays_count, &system->reduce_result_int[1], n_faces);

    int n_rays_count[2];
    cudaMemcpyAsync(n_rays_count, system->reduce_result_int, 2 * sizeof(int), cudaMemcpyDeviceToHost, system->memcpy_stream);

    // Grow task
    int n_upper_bound = system->params.n_rays + 2 * n_faces;
    int n_input_rays = task_grow(task, n_upper_bound);

    // Generate random
    random_ctx_t random_ctx = make_random_ctx(system);
    generate_malley_random << <system->n_mtgp_blocks, 256 >> > (random_ctx, system->reduce_result_int, (ray_t*)task->rays->ray + n_input_rays);

    cudaStreamSynchronize(system->memcpy_stream);
    int n_front_rays = n_rays_count[0];
    int n_rear_rays = n_rays_count[1];
    int n_real_rays = n_front_rays + n_rear_rays;
    task->rays->n_tasks -= n_upper_bound - n_real_rays;

    // Scale face ray emission by rays count
    g = system->scale_emission_grid;
    scale_emission << <g.grid(n_faces), g.block(n_faces) >> > (n_faces,
      system->face_front_rays_count,
      system->face_front_emission
      );
    scale_emission << <g.grid(n_faces), g.block(n_faces) >> > (n_faces,
      system->face_rear_rays_count,
      system->face_rear_emission
      );

    // Emit rays
    if (n_front_rays)
    {
      int n_block_size = std::min(1024, (n_front_rays / n_faces + 31) & -32);
      cub::DeviceScan::ExclusiveSum(system->reduce_buffer, system->reduce_buffer_size, system->face_front_rays_count, system->face_rays_offset, n_faces);
      emit_malley_rays<true> << <n_faces, n_block_size >> > (scene_ctx,
        system->face_front_rays_count,
        system->face_rays_offset,
        0,
        system->face_front_emission,
        (ray_t*)task->rays->ray + n_input_rays,
        task->rays->power + n_input_rays
        );
    }
    if (n_rear_rays)
    {
      int n_block_size = std::min(1024, (n_rear_rays / n_faces + 31) & -32);
      cub::DeviceScan::ExclusiveSum(system->reduce_buffer, system->reduce_buffer_size, system->face_rear_rays_count, system->face_rays_offset, n_faces);
      emit_malley_rays<false> << <n_faces, n_block_size>> > (scene_ctx,
        system->face_rear_rays_count,
        system->face_rays_offset,
        n_front_rays,
        system->face_rear_emission,
        (ray_t*)task->rays->ray + n_input_rays,
        task->rays->power + n_input_rays
        );
    }

    // Update mesh emission
    if (r = scene_relocate(system->scene, -1))
      return r;
    malley_cpu::update_mesh_emission(system->scene, task);

    return EMISSION_OK;
  }

  int init(system_t* system, const malley_cpu::params_t* params)
  {
    system->params = *params;

    system->scene = 0;
    system->kernel_params = 0;

    int n_mtgp_blocks = 128;
    system->n_mtgp_blocks = n_mtgp_blocks;

    checkCudaErrors(cudaMalloc((void **)&system->GenA, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&system->GenB, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&system->GenR, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&system->GenTheta, n_mtgp_blocks * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void **)&system->kernel_params, sizeof(mtgp32_kernel_params)));
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, system->kernel_params));
    CURAND_CALL(curandMakeMTGP32KernelState(system->GenA, mtgp32dc_params_fast_11213, system->kernel_params,
      n_mtgp_blocks, 1));
    CURAND_CALL(curandMakeMTGP32KernelState(system->GenB, mtgp32dc_params_fast_11213, system->kernel_params,
      n_mtgp_blocks, 2));
    CURAND_CALL(curandMakeMTGP32KernelState(system->GenR, mtgp32dc_params_fast_11213, system->kernel_params,
      n_mtgp_blocks, 3));
    CURAND_CALL(curandMakeMTGP32KernelState(system->GenTheta, mtgp32dc_params_fast_11213, system->kernel_params,
      n_mtgp_blocks, 4));

    system->calculate_weights_grid = grid_calculator_t(calculate_weights);
    system->scale_emission_grid = grid_calculator_t(scale_emission);
    system->calculate_rays_count_grid = grid_calculator_t(calculate_rays_count);

    checkCudaErrors(cudaStreamCreate(&system->memcpy_stream));

    return EMISSION_OK;
  }

  int set_scene(system_t* system, subject::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    if (scene->n_meshes == 0)
      return -EMISSION_ERROR;

    if (scene->n_materials == 0)
      return -EMISSION_ERROR;

    system->scene = scene;

    checkCudaErrors(cudaMalloc((void**)&system->mesh_temperature, scene->n_meshes * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&system->face_front_weight, scene->n_faces * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&system->face_rear_weight, scene->n_faces * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&system->face_front_emission, scene->n_faces * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&system->face_rear_emission, scene->n_faces * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&system->face_front_rays_count, scene->n_faces * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&system->face_rear_rays_count, scene->n_faces * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&system->face_rays_offset, scene->n_faces * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&system->face_rays_count_next, scene->n_faces * sizeof(int)));

    system->reduce_buffer_size = std::max(size_t(4096), scene->n_faces * sizeof(int));
    checkCudaErrors(cudaMalloc((void**)&system->reduce_buffer, system->reduce_buffer_size));
    checkCudaErrors(cudaMalloc((void**)&system->reduce_result_float, 2 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&system->reduce_result_int, 2 * sizeof(int)));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return EMISSION_OK;
  }

  int shutdown(system_t* system)
  {
    system->scene = 0;

    cudaFree(system->GenA);
    cudaFree(system->GenB);
    cudaFree(system->GenR);
    cudaFree(system->GenTheta);

    cudaFree(system->kernel_params);
    system->kernel_params = 0;

    cudaFree(system->mesh_temperature);
    cudaFree(system->face_front_weight);
    cudaFree(system->face_rear_weight);
    cudaFree(system->face_front_emission);
    cudaFree(system->face_rear_emission);
    cudaFree(system->face_front_rays_count);
    cudaFree(system->face_rear_rays_count);
    cudaFree(system->face_rays_offset);
    cudaFree(system->face_rays_count_next);
    cudaFree(system->reduce_buffer);
    cudaFree(system->reduce_result_int);
    cudaFree(system->reduce_result_float);
    system->mesh_temperature = 0;
    system->face_front_weight = 0;
    system->face_rear_weight = 0;
    system->face_front_emission = 0;
    system->face_rear_emission = 0;
    system->face_front_rays_count = 0;
    system->face_rear_rays_count = 0;
    system->face_rays_offset = 0;
    system->face_rays_count_next = 0;
    system->reduce_buffer = 0;
    system->reduce_result_int = 0;
    system->reduce_result_float = 0;

    cudaStreamDestroy(system->memcpy_stream);

    return EMISSION_OK;
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
    system_t* s = (system_t*)calloc(1, sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
