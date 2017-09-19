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

#include <map>
#include <thrust/system/cuda/vector.h>

#include <helper_cuda.h>
namespace {
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
}

namespace {
  class cached_allocator
  {
  public:
    // just allocate bytes
    typedef char value_type;

    cached_allocator() {}

    ~cached_allocator()
    {
      // free all allocations when cached_allocator goes out of scope
      free_all();
    }

    char *allocate(std::ptrdiff_t num_bytes)
    {
      char *result = 0;

      // search the cache for a free block
      free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

      if (free_block != free_blocks.end())
      {
        // get the pointer
        result = free_block->second;

        // erase from the free_blocks map
        free_blocks.erase(free_block);
      }
      else
      {
        // no allocation of the right size exists
        // create a new one with cuda::malloc
        // throw if cuda::malloc can't satisfy the request
        try
        {
          // allocate memory and convert cuda::pointer to raw pointer
          result = thrust::cuda::malloc<char>(num_bytes).get();
        }
        catch (std::runtime_error &e)
        {
          throw;
        }
      }

      // insert the allocated pointer into the allocated_blocks map
      allocated_blocks.insert(std::make_pair(result, num_bytes));

      return result;
    }

    void deallocate(char *ptr, size_t n)
    {
      // erase the allocated block from the allocated blocks map
      allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
      std::ptrdiff_t num_bytes = iter->second;
      allocated_blocks.erase(iter);

      // insert the block into the free blocks map
      free_blocks.insert(std::make_pair(num_bytes, ptr));
    }

  private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char *, std::ptrdiff_t>     allocated_blocks_type;

    free_blocks_type      free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all()
    {
      // deallocate all outstanding blocks in both lists
      for (free_blocks_type::iterator i = free_blocks.begin();
        i != free_blocks.end();
        ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
      }

      for (allocated_blocks_type::iterator i = allocated_blocks.begin();
        i != allocated_blocks.end();
        ++i)
      {
        // transform the pointer to cuda::pointer before calling cuda::free
        thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
      }
    }

  };
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
    int n_postpone_block_size;

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

    cached_allocator* thrust_allocator;

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

    delete system->thrust_allocator;

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
    hit_diffuse_front = 2,
    hit_diffuse_rear = 3,
    hit_transmit = 4,
    hit_drop = 5,
    hit_missed = 6,
  };

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

  __device__ void inplace_warp_scan(int* in)
  {
    int n = 32;

    __shared__ int temp[32];
    int tid = threadIdx.x;
    int offset = 1;

    // A
    temp[2*thid] = in[2*thid]; // load input into shared memory
    temp[2*thid+1] = in[2*thid+1];

    for (int d = 16; d > 0; d >>= 1)                    // build sum in place up the tree
    {
      __syncthreads();
      if (thid < d)
      {
        //B
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        temp[bi] += temp[ai];
      }
      offset *= 2;

      // C
      if (thid == 0) { temp[n - 1] = 0; } // clear the last element
      for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
      {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {

          // D
          int ai = offset * (2 * thid + 1) - 1;
          int bi = offset * (2 * thid + 2) - 1;
          int t = temp[ai];
          temp[ai] = temp[bi];
          temp[bi] += t;
        }
      }
      __syncthreads();
    }
  }


  __global__ void sort_by_interaction(int n_rays, const char* hit_decision, const int* thread_offset, int n_diffused_front_offset, int n_diffused_rear_offset, int n_reflected_offset, int n_transmitted_offset,
                                      const ray_t* s_ray, const float3* s_hit_point, const float* s_power, const int* s_hit_face_idx,
                                      ray_t* t_ray, float3* t_hit_point, float* t_power, int* t_hit_face_idx
  )
  {
    const int n_interactions = 5;

    int skips[n_interactions];
    ray_t ray_buffer;
    float3 hit_point_buffer;
    float power_buffer;
    int hit_face_buffer;

    __shared__ int warp_progress[32];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; id < n_rays; id += stride)
    {
      const char decision = hit_decision[id];
      const bool skip = decision == hit_missed || decision == hit_drop;

      ray_buffer = s_ray[id];
      hit_point_buffer = s_hit_point[id];
      power_buffer = s_power[id];
      hit_face_buffer = s_hit_face_idx[id];

      for (int i = 0; i != n_interactions; ++i)
      {
        int ballot = __ballot(i == decision);
        int warp_thread_idx = threadIdx.x % warpSize;
        int mask = ~int(0) >> (warpSize - warp_thread_idx - 1);
        int warp_offset = __popc(ballot & mask);

        int warp_idx = threadIdx.x / warpSize; // @todo Add if?
        warp_progress[warp_idx] = __popc(ballot);
      }


      if (flush_thread_idx[decision] == threadIdx.x)
      {
        int expected = next_buffered[decision] - buffered[decision];
        int write_offset = atomicAdd(&global_offsets[decision], buffered[decision]);
        for (int i = 0; i != buffered[decision]; ++i)
        {
          t_ray[write_offset + i] = ray_buffer[i];
          t_hit_point[write_offset + i] = hit_point_buffer[i];
          t_power[write_offset + i] = power_buffer[i];
          t_hit_face_idx[write_offset + i] = hit_face_buffer[i];
        }
        buffered[decision] = expected;
        next_buffered[decision] = expected;
      }
      __syncthreads();

      if (!skip)
      {
        int buf_idx = atomicAdd(&buffered[decision], 1);
        ray_buffer[buf_idx] = s_ray[id];
        hit_point_buffer[buf_idx] = s_hit_point[id];
        power_buffer[buf_idx] = s_power[id];
        hit_face_buffer[buf_idx] = s_hit_face_idx[id];
      }
    }
  }

  __global__ void ray_absorption(int n_rays, const int* hit_face_idx, const int* face_to_mesh, const float* ray_power, float* mesh_power)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      int face_idx = hit_face_idx[ray_idx];
      int mesh_idx = face_to_mesh[face_idx];
      atomicAdd(&mesh_power[mesh_idx], ray_power[ray_idx]);
    }
  }

  template <bool front>
  __global__ void ray_diffusion(context_t* ctx, int n_rays, const int* hit_face_idx, const float3* hit_points, const float3* face_normals, ray_t* rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      int face_idx = hit_face_idx[ray_idx];
      float3 hit_point = hit_points[ray_idx];
      float3 face_normal = face_normals[face_idx];
      float3 malley_ray = malley_cuda::pick_uniform_ray(ctx);
      
      if (!front)
        malley_ray = cuda_math::invert_z(malley_ray);

      rays[ray_idx] = malley_cuda::ray_malley_emmit(malley_ray, face_normal, hit_point);
    }
  }

  __global__ void ray_reflection(int n_rays, const int* hit_face_idx, const float3* hit_points, const float3* face_normals, ray_t* rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      const ray_t& ray = rays[ray_idx];
      int face_idx = hit_face_idx[ray_idx];
      float3 hit_point = hit_points[ray_idx];
      float3 face_normal = face_normals[face_idx];

      rays[ray_idx] = cuda_math::ray_reflect(ray, face_normal, hit_point);
    }
  }

  __global__ void ray_transmition(int n_rays, const float3* hit_points, ray_t* rays)
  {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; ray_idx < n_rays; ray_idx += stride)
    {
      const ray_t& ray = rays[ray_idx];
      float3 hit_point = hit_points[ray_idx];
      
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

    system->thrust_allocator = new cached_allocator();

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

    system->n_postpone_blocks = 64;
    system->n_postpone_block_size = 128;
    
    checkCudaErrors(cudaMalloc((void **)&host_context.postponed_offsets, sizeof(int) * n_interaction_types_count * system->n_postpone_blocks * system->n_postpone_block_size));

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

    ray_caster::task_t* ray_caster_task = emission_task->rays;
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
    make_ray_decisions<<<system->n_postpone_blocks, system->n_postpone_block_size>>>(n_rays,
                        ray_caster_task->hit_face_idx,
                        (ray_t*)ray_caster_task->ray,
                        system->host_context.ray_random,
                        (const float3*)scene->face_normals,
                        scene->face_to_mesh,
                        scene->materials,
                        scene->mesh_material,
                        system->host_context.hit_decision,
                        system->host_context.postponed_offsets);
    const int n_reducer_stride = system->n_postpone_blocks * system->n_postpone_block_size;
    thrust::device_ptr<int> postponed_offsets(system->host_context.postponed_offsets);

    interaction_counts_t n_counts;
    for (int i = 0; i != n_interaction_types_count; ++i)
    {
      n_counts.n[i] = thrust::reduce(thrust::cuda::par(*system->thrust_allocator),
                                     postponed_offsets + i * n_reducer_stride,
                                     postponed_offsets + (i + 1) * n_reducer_stride);
      thrust::exclusive_scan(thrust::cuda::par(*system->thrust_allocator), postponed_offsets + i * n_reducer_stride,
                             postponed_offsets + (i + 1) * n_reducer_stride, postponed_offsets + i * n_reducer_stride);
    }

    source_interaction_offsets_t n_source_offsets(n_counts);
    target_interaction_offsets_t n_target_offsets(n_counts);

    const ray_caster::task_t* source = ray_caster_task;
    ray_caster::task_t* target = postponed_task->rays;
    sort_by_interaction<<<system->n_postpone_blocks, system->n_postpone_block_size>>>(n_rays,
            system->host_context.hit_decision,
            system->host_context.postponed_offsets,
            n_source_offsets.named.n_diffused_front,
            n_source_offsets.named.n_diffused_rear,
            n_source_offsets.named.n_reflected,
            n_source_offsets.named.n_transmitted,
            (ray_t*)source->ray,
            (const float3*)source->hit_point,
            source->power,
            source->hit_face_idx,
            (ray_t*)target->ray,
            (float3*)target->hit_point,
            target->power,
            target->hit_face_idx
    );

    int n_postponed = n_counts.postponed();
    const int n_missed = n_rays - n_counts.named.n_absorbed - n_postponed;

    // @detail Yet alignment it is not crucial for ray index resolving.
    // @todo Does align of interaction groups on cache-line boundary speedup further interaction kernel application?
    // @todo Without alignment we can run single postpone kernel.

    if (n_counts.named.n_absorbed)
    {
      const int n_ray_absorption_grid = (n_counts.named.n_absorbed + system->block_size.n_ray_absorption - 1) / system->block_size.n_ray_absorption;
      ray_absorption << <n_ray_absorption_grid, system->block_size.n_ray_absorption >> > (
        n_counts.named.n_absorbed,
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
        (const float3*)ray_caster_task->hit_point,
        (ray_t*)ray_caster_task->ray
        );
    }

    cudaMemcpy(source->ray, target->ray, sizeof(ray_t) * n_postponed, cudaMemcpyDeviceToDevice);
    cudaMemcpy(source->power, target->power, sizeof(float) * n_postponed, cudaMemcpyDeviceToDevice);

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
          (const ray_t*)ray_caster_task->ray + n_source_offset,
          ray_caster_task->power + n_source_offset,
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
