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
#include "system.cuh"

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

#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__); \
  return EXIT_FAILURE;}} while(0)

namespace cuda_malley_emission
{
  using namespace cuda_math;

  /// @brief Extended base system_t (C-style polymorphism)
  struct system_t : emission::system_t
  {
    params_t params;

    subject::scene_t* scene;
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

  /// @brief Initializes system with given ray caster after creation.
  int init(system_t* system, const malley_cpu::params_t* params)
  {
    system->params = *params;

    system->scene = 0;
    system->host_context = {};
    system->device_context = 0;
    system->kernel_params = 0;

    const char* argv[] = { "" };
    int device_id = findCudaDevice(1, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_id));
    system->n_tpb = deviceProp.maxThreadsPerBlock;
    system->n_warp_size = deviceProp.warpSize;
    //system->n_tpb = 128;

    int n_mtgp_blocks = 64;
    system->n_mtgp_blocks = n_mtgp_blocks;

    context_t& host_context = system->host_context;
    checkCudaErrors(cudaMalloc((void **)&host_context.TPGenA, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.TPGenB, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenR, n_mtgp_blocks * sizeof(curandStateMtgp32)));
    checkCudaErrors(cudaMalloc((void **)&host_context.HSGenTheta, n_mtgp_blocks * sizeof(curandStateMtgp32)));

    checkCudaErrors(cudaMalloc((void**)&system->kernel_params, sizeof(mtgp32_kernel_params)));
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, system->kernel_params));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.TPGenA, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 1));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.TPGenB, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 2));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenR, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 3));
    CURAND_CALL(curandMakeMTGP32KernelState(host_context.HSGenTheta, mtgp32dc_params_fast_11213, system->kernel_params, n_mtgp_blocks, 4));

    return EMISSION_OK;
  }

  void contex_free_scene(context_t* ctx)
  {
    if (ctx->faces)
      cudaFree(ctx->faces);
    ctx->faces = 0;

    if (ctx->faces_data)
      cudaFree(ctx->faces_data);
    ctx->faces_data = 0;

    if (ctx->materials)
      cudaFree(ctx->materials);
    ctx->materials = 0;
    
    if (ctx->mesh_materials)
      cudaFree(ctx->mesh_materials);
    ctx->mesh_materials = 0;

    if (ctx->mesh_emission)
      cudaFree(ctx->mesh_emission);
    ctx->mesh_emission = 0;

    if (ctx->ray_power)
      cudaFree(ctx->ray_power);
    ctx->ray_power = 0;

    if (ctx->rays)
      cudaFree(ctx->rays);
    ctx->rays = 0;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(system_t* system)
  {
    system->scene = 0;

    contex_free_scene(&system->host_context);
    
    cudaFree(system->host_context.TPGenA);
    cudaFree(system->host_context.TPGenB);
    cudaFree(system->host_context.HSGenR);
    cudaFree(system->host_context.HSGenTheta);

    system->host_context = {};
    cudaFree(system->device_context);
    system->device_context = 0;

    cudaFree(system->kernel_params);
    system->kernel_params = 0;

    return EMISSION_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(system_t* system, subject::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    if (scene->n_meshes == 0)
      return -EMISSION_ERROR;

    if (scene->n_materials == 0)
      return -EMISSION_ERROR;

    system->scene = scene;

    context_t& host_context = system->host_context;
    contex_free_scene(&host_context);

    host_context.n_faces = scene->n_faces;

    // Init device faces
    checkCudaErrors(cudaMalloc((void**)&host_context.faces, scene->n_faces * sizeof(face_t)));
    checkCudaErrors(cudaMemcpy(host_context.faces, scene->faces, scene->n_faces * sizeof(face_t), cudaMemcpyHostToDevice));

    // Init device faces data
    checkCudaErrors(cudaMalloc((void**)&host_context.faces_data, scene->n_faces * sizeof(face_data_t)));
    checkCudaErrors(cudaMalloc((void**)&host_context.face_ray_offsets, scene->n_faces * sizeof(int)));

    // Init device materials
    checkCudaErrors(cudaMalloc((void**)&host_context.materials, scene->n_materials * sizeof(subject::material_t)));
    checkCudaErrors(cudaMemcpy(host_context.materials, scene->materials, scene->n_materials * sizeof(subject::material_t), cudaMemcpyHostToDevice));

    // Init meshes materials. @todo Store meshes instead?
    checkCudaErrors(cudaMalloc((void**)&host_context.mesh_materials, scene->n_meshes * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&host_context.mesh_emission, scene->n_meshes * sizeof(float)));

    sync_context_to_device(system);

    // Temp copy of meshes to init context
    subject::mesh_t* device_meshes;
    checkCudaErrors(cudaMalloc((void**)&device_meshes, scene->n_meshes * sizeof(subject::mesh_t)));    
    checkCudaErrors(cudaMemcpy(device_meshes, scene->meshes, scene->n_meshes * sizeof(subject::mesh_t), cudaMemcpyHostToDevice));
    
    int n_thread_per_mesh = system->n_warp_size;
    int n_meshes_per_block = system->n_tpb / n_thread_per_mesh;
    dim3 threads(n_meshes_per_block, n_thread_per_mesh);
    int n_blocks = (scene->n_meshes + n_meshes_per_block - 1) / n_meshes_per_block;
    prepare_context << <n_blocks, threads >> >(system->device_context, device_meshes, scene->n_meshes);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaFree(device_meshes);

    return EMISSION_OK;
  }

  __global__ void prepare_context(context_t* context, const subject::mesh_t* meshes, int n_meshes)
  {
    face_data_t* faces_data = context->faces_data;

    int mesh_idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (; mesh_idx < n_meshes; mesh_idx += gridDim.x * blockDim.x)
    {
      const subject::mesh_t& mesh = meshes[mesh_idx];
      context->mesh_materials[mesh_idx] = mesh.material_idx;

      int mesh_face_idx = threadIdx.y;      
      for (; mesh_face_idx < mesh.n_faces; mesh_face_idx += blockDim.y)
      {        
        int face_idx = mesh.first_idx + mesh_face_idx;
        face_t& face = context->faces[face_idx];
        convert_face_to_uv(face);
        float area = face_area(face);

        face_data_t& data = faces_data[face_idx];
        data = { mesh_idx , area };
      }
    }
  }

  #define sigma 5.670400e-8f

  __global__ void calculate_weights(context_t* context, const float* temperatures)
  {
    int face_idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    int n_faces = context->n_faces;
    face_data_t* faces_data = context->faces_data;
    subject::material_t* materials = context->materials;
    int* mesh_materials = context->mesh_materials;

    for (; face_idx < n_faces; face_idx += gridDim.x * blockDim.x)
    {
      face_data_t& data = faces_data[face_idx];

      int mesh_idx = data.mesh_idx;
      if (mesh_idx >= 0)
      { 
        float area = data.area;
        const subject::material_t& material = materials[mesh_materials[mesh_idx]];
        float2 potential = area * make_float2(material.front.emissivity, material.rear.emissivity);        

        float T = temperatures[mesh_idx];
        float weight_density = T; // some empirical distribution function of temperature
        float power_density = sigma * (T * T * T * T);

        data.weights = weight_density * potential;
        data.emission = power_density * potential;
      }
    }
  }

  // gridDim.x is not required - running single block only!
  __global__ void calculate_n_rays(context_t* ctx, int n_target_rays)
  {
    extern __shared__ float2 weights[];
    extern __shared__ int n_total_rays[];

    int n_faces = ctx->n_faces;
    int face_idx = threadIdx.x;
    int thread_idx = threadIdx.x;
    face_data_t* faces_data = ctx->faces_data;

    weights[thread_idx] = {};

    for (; face_idx < n_faces; face_idx += blockDim.x)
      weights[thread_idx] += faces_data[face_idx].weights;

    __syncthreads();

    int m = blockDim.x;
    while (m > 1)
    {
      int half_m = m >> 1;
      if (thread_idx < half_m)
        weights[thread_idx] += weights[m - thread_idx - 1];
      __syncthreads();

      m -= half_m;
    }

    if (thread_idx == 0)
      ctx->weight_factor = 1.f / (weights[0].x + weights[0].y);

    __syncthreads();

    n_total_rays[thread_idx] = {};
    float weight_factor = ctx->weight_factor;

    int* face_ray_offsets = ctx->face_ray_offsets;
    face_idx = threadIdx.x;
    for (; face_idx < n_faces; face_idx += blockDim.x)
    {
      face_data_t& data = faces_data[face_idx];
      int2 n_face_rays = make_int2(ceilf(n_target_rays * weight_factor * data.weights));
      data.n_rays = n_face_rays;
      int n_total_face_rays = n_face_rays.x + n_face_rays.y;
      n_total_rays[thread_idx] += n_total_face_rays;
      face_ray_offsets[face_idx] = n_total_face_rays;
    }

    __syncthreads();

    m = blockDim.x;
    while (m > 1)
    {
      int half_m = m >> 1;
      if (thread_idx < half_m)
        n_total_rays[thread_idx] += n_total_rays[m - thread_idx - 1];
      __syncthreads();

      m -= half_m;
    }

    if (thread_idx == 0)
      ctx->n_total_rays = n_total_rays[0];
}

  __device__ float3 pick_face_point(context_t* ctx, const face_t& face)
  { 
    float a = curand_uniform(&ctx->TPGenA[blockIdx.x]);
    float b = curand_uniform(&ctx->TPGenB[blockIdx.x]);
    if (a + b > 1)
    {
      a = 1.f - a;
      b = 1.f - b;
    }

    return face.points[0] + a * face.points[1] + b * face.points[2];
  }

  __device__ float3 pick_uniform_ray(context_t* ctx)
  {
    float r = curand_uniform(&ctx->HSGenR[blockIdx.x]);
    float theta = M_2PI * curand_uniform(&ctx->HSGenTheta[blockIdx.x]);
    
    float rad = sqrtf(r);
    float s, c;
    sincos(theta, &s, &c);
    return make_float3(rad * c, rad * s, sqrtf(1 - r));
  }

  __global__ void emit_malley_rays(context_t* ctx, ray_t* rays, float* mesh_emissions, float* rays_power)
  {
    int face_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int n_faces = ctx->n_faces;

    for (; face_idx < n_faces; face_idx += blockDim.x * gridDim.x)
    {
      face_t face = ctx->faces[face_idx];
      const face_data_t& face_data = ctx->faces_data[face_idx];
      int2 n_rays = face_data.n_rays;

      float2 emission = face_data.emission;
      mesh_emissions[face_data.mesh_idx] += emission.x + emission.y;

      mat33 rotation = pick_face_rotation(face);
      
      float ray_power = emission.x / (float)n_rays.x;
      int face_ray_offset = ctx->face_ray_offsets[face_idx];
      for (int j = threadIdx.y; j < n_rays.x; j += blockDim.y)
      {
        rays_power[face_ray_offset + j] = ray_power;
        float3 relative_dist = rotation * pick_uniform_ray(ctx);
        rays[face_ray_offset + j] = { pick_face_point(ctx, face) + relative_dist * 0.0001f, relative_dist };
      }

      ray_power = emission.y / (float)n_rays.y;
      face_ray_offset += face_data.n_rays.x;
      for (int j = threadIdx.y; j < n_rays.y; j += blockDim.y)
      {
        rays_power[face_ray_offset + j] = ray_power;
        float3 relative_dist = rotation * invert_z(pick_uniform_ray(ctx));
        rays[face_ray_offset + j] = { pick_face_point(ctx, face) + relative_dist * 0.0001f, relative_dist };
      }
    }
  }

  int next_power_of_2(int v)
  {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  }

  int prev_power_of_2(int v)
  { 
    return next_power_of_2(v) >> 1; 
  }

  void calculate_weights(system_t* system, emission::task_t* task)
  {
    subject::scene_t* scene = system->scene;
    // @todo What is optimal block size?
    int n_threads = std::min<int>(system->n_warp_size, scene->n_meshes);
    int n_blocks = (system->scene->n_faces + n_threads - 1) / n_threads;
    thrust::device_vector<float> device_temperatures(task->mesh_temperatures, task->mesh_temperatures + scene->n_meshes);

    calculate_weights << <n_blocks, n_threads >> >(system->device_context, thrust::raw_pointer_cast(device_temperatures.data()));

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void calculate_n_rays(system_t* system)
  {
    subject::scene_t* scene = system->scene;
    int shared_size = system->n_tpb * (sizeof(float2) + sizeof(int));

    // Using shared memory for reductions. So run single block of maximum size.
    calculate_n_rays << <1, system->n_tpb, shared_size >> > (system->device_context, system->params.n_rays);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::device_ptr<int> ray_offsets(system->host_context.face_ray_offsets);
    thrust::exclusive_scan(ray_offsets, ray_offsets + scene->n_faces, ray_offsets);
    cudaMemcpy(&system->host_context, system->device_context, sizeof(context_t), cudaMemcpyDeviceToHost);
  }

  void grow_device_buffers(system_t* system)
  {
    context_t& host_context = system->host_context;
    int n_rays = host_context.n_total_rays;

    if (host_context.n_ray_buffer < n_rays)
    {
      if (host_context.ray_power)
      {
        cudaFree(host_context.ray_power);
        cudaFree(host_context.rays);
      }

      int n_new_rays = n_rays + 1024 - n_rays % 1024;
      host_context.n_ray_buffer = n_new_rays;
      cudaMalloc(&host_context.ray_power, n_new_rays * sizeof(float));
      cudaMalloc(&host_context.rays, n_new_rays * sizeof(ray_t));

      sync_context_to_device(system);
    }
  }

  int calculate(system_t* system, emission::task_t* task)
  {
    calculate_weights(system, task);
    calculate_n_rays(system);

    int n_real_rays = system->host_context.n_total_rays;

    if (n_real_rays == 0)
      return EMISSION_OK;

    grow_device_buffers(system);

    int n_input_rays = task_grow(task, n_real_rays);

    context_t& host_context = system->host_context;
    subject::scene_t* scene = system->scene;
    cudaMemcpy(host_context.mesh_emission, task->mesh_emission, scene->n_meshes * sizeof(float), cudaMemcpyHostToDevice);

    int n_avg_rays_per_face = prev_power_of_2(system->params.n_rays / scene->n_faces);
    if (n_avg_rays_per_face < 2)
      n_avg_rays_per_face = 2;
    // Run multiple threads for faces with large amount of rays.
    //int n_threads_per_face = 1;
    int n_threads_per_face = std::min<int>(32, n_avg_rays_per_face);
    
    dim3 threads(256 / n_threads_per_face, n_threads_per_face);
    emit_malley_rays << <system->n_mtgp_blocks, threads >> > (system->device_context
      , host_context.rays
      , host_context.mesh_emission
      , host_context.ray_power
      );

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Using raw memory copy because of incompatible data-types.
    cudaMemcpy(&task->rays->ray[n_input_rays], host_context.rays, n_real_rays * sizeof(ray_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&task->power[n_input_rays], host_context.ray_power, n_real_rays * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(task->mesh_emission, host_context.mesh_emission, scene->n_meshes * sizeof(float), cudaMemcpyDeviceToHost);

    return EMISSION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
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
    host_context.n_allocated = calculate_total_rays(system->params);
    checkCudaErrors(cudaMalloc((void **)&host_context.rays, host_context.n_allocated * sizeof(ray_t)));
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
    if (system->host_context.rays)
      cudaFree(system->host_context.rays);
    system->host_context.rays = 0;

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

  __global__ void emit_parallel_rays(context_t* ctx, source_params_t params, ray_t* rays)
  { 
    float3 origin = ctx->origin;
    float radius = ctx->radius;

    float offset = 1.3f + radius;
    const mat33 rotation = rotate_towards(make_float3(0, 0, 1), params.direction);
    float3 direction = params.direction;
    int n_rays = params.n_rays;

    curandStateMtgp32* HSGenR = &ctx->HSGenR[blockIdx.x];
    curandStateMtgp32* HSGenTheta = &ctx->HSGenTheta[blockIdx.x];
    
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; ray_idx < n_rays; ray_idx += blockDim.x * gridDim.x)
    {
      float r = radius * sqrtf(curand_uniform(HSGenR));
      float theta = M_2PI * curand_uniform(HSGenTheta);

      float s, c;
      sincos(theta, &s, &c);

      float3 ray_origin = rotation * make_float3(r * c, r * s, -offset) + origin;
      rays[ray_idx] = { ray_origin, direction };
    }
  }

  int calculate(system_t* system, emission::task_t* task)
  {
    align_rays_count_to_warp_size(system->params, system->n_warp_size);
    int n_rays = calculate_total_rays(system->params);
    context_t& host_context = system->host_context;
    if (host_context.n_allocated != n_rays)
    {
      checkCudaErrors(cudaFree(host_context.rays));
      host_context.n_allocated = n_rays;
      checkCudaErrors(cudaMalloc((void **)&host_context.rays, host_context.n_allocated * sizeof(ray_t)));
    }

    int n_total_rays = calculate_total_rays(system->params);
    
    ray_t* device_rays = system->host_context.rays;
    const int n_input_rays = task_grow(task, n_total_rays);
    const int n_sources = system->params.n_sources;

    // Launch kernels
    for (int source_idx = 0; source_idx != n_sources; ++source_idx)
    {
      parallel_rays_emission_cpu::parallel_rays_source_t& source = system->params.sources[source_idx];
      if (source.n_rays == 0)
        continue;

      source_params_t params = { make_float3(source.direction.x, source.direction.y, source.direction.z), source.n_rays };
      emit_parallel_rays << <system->n_mtgp_blocks, 256 >> > (system->device_context, params, device_rays);
      device_rays += source.n_rays;
    }

    // Fill ray powers
    int n_host_rays_offset = n_input_rays;
    float radius = system->host_context.radius;
    for (int source_idx = 0; source_idx != n_sources; ++source_idx)
    {
      parallel_rays_emission_cpu::parallel_rays_source_t& source = system->params.sources[source_idx];
      const float ray_power = source.power * (float(M_PI) * radius * radius) / (float)source.n_rays;
      for (int r = 0; r != source.n_rays; ++r)
        task->power[n_host_rays_offset + r] = ray_power;
      n_host_rays_offset += source.n_rays;
    }

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Using raw memory copy because of incompatible data-types.
    cudaMemcpy(&task->rays->ray[n_input_rays], system->host_context.rays, n_total_rays * sizeof(ray_t), cudaMemcpyDeviceToHost);

    return EMISSION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
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
