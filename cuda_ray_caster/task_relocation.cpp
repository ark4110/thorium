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

#include "task_relocation.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include "../cuda_misc/cuda_check_errors.h"

namespace cuda_task_relocation
{
  void cuda_rays_data_free(ray_caster::rays_data_t* data)
  {
    cudaFree(data->ray);
    cudaFree(data->hit_distance);
    cudaFree(data->hit_face_idx);
    cudaFree(data->power);
  }

  void cuda_rays_data_alloc(int n_alloc, ray_caster::rays_data_t* data)
  {
    checkCudaErrors(cudaMalloc((void**)&data->ray, sizeof(math::ray_t) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&data->hit_distance, sizeof(float) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&data->hit_face_idx, sizeof(int) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&data->power, sizeof(float) * n_alloc));
  }

  void cuda_rays_data_copy_all(int n_tasks, ray_caster::rays_data_t* destination, ray_caster::rays_data_t* source, cudaMemcpyKind direction)
  {
    if (!n_tasks)
      return;
    checkCudaErrors(cudaMemcpy(destination->ray, source->ray, n_tasks * sizeof(math::ray_t), direction));
    checkCudaErrors(cudaMemcpy(destination->hit_distance, source->hit_distance, n_tasks * sizeof(float), direction));
    checkCudaErrors(cudaMemcpy(destination->hit_face_idx, source->hit_face_idx, n_tasks * sizeof(int), direction));
    checkCudaErrors(cudaMemcpy(destination->power, source->power, n_tasks * sizeof(float), direction));
  }

  void cuda_task_free_data(ray_caster::task_t* task)
  {
    cuda_rays_data_free(task);
    cuda_rays_data_free(&task->next_gen);
  }

  int grow_task(ray_caster::task_t* task, int n_rays)
  {
    int n_prev = task->n_tasks;
    if (task->n_tasks + n_rays < task->n_reserved)
    {
      task->n_tasks += n_rays;
      return n_prev;
    }

    ray_caster::task_t* buffer_task = ray_caster::task_create(task->n_tasks);

    cuda_rays_data_copy_all(task->n_tasks, buffer_task, task, cudaMemcpyDeviceToHost);
    // We grow to create new rays -> no need to preserve hit_faces_idx and hit_point.
    if (task->next_gen.n_tasks)
    {
      checkCudaErrors(cudaMemcpy(buffer_task->next_gen.ray, task->next_gen.ray, task->next_gen.n_tasks * sizeof(math::ray_t), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(buffer_task->next_gen.power, task->next_gen.power, task->next_gen.n_tasks * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cuda_rays_data_free(task);
    cuda_rays_data_free(&task->next_gen);

    ray_caster::task_t rtask = {};
    rtask.allocation_device = task->allocation_device;
    rtask.free_data = &cuda_task_free_data;
    rtask.grow = &grow_task;
    rtask.relocate = &relocate_task;
    
    rtask.n_tasks = task->n_tasks + n_rays;
    const int align_factor = pow(2, ceil(log(n_rays)/log(2))) / 2;
    int n_alloc = (rtask.n_tasks + align_factor - 1) & -align_factor;
    rtask.n_reserved = n_alloc;

    cuda_rays_data_alloc(n_alloc, &rtask);
    cuda_rays_data_alloc(n_alloc, &rtask.next_gen);
    
    cuda_rays_data_copy_all(task->n_tasks, &rtask, buffer_task, cudaMemcpyHostToDevice);
    if (task->next_gen.n_tasks)
    {
      checkCudaErrors(cudaMemcpy(rtask.next_gen.ray, buffer_task->next_gen.ray, task->next_gen.n_tasks * sizeof(math::ray_t), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(rtask.next_gen.power, buffer_task->next_gen.power, task->next_gen.n_tasks * sizeof(float), cudaMemcpyHostToDevice));
    }
    task_free(buffer_task);

    if (rtask.n_reserved > 128*1024)
      fprintf(stderr, "Task grow from %d to %d rays\n", n_prev, rtask.n_reserved);

    *task = rtask;

    return n_prev;
  }

  int relocate_task(ray_caster::task_t* task, int device)
  {
    if (!task)
      return RAY_CASTER_OK;

    if (task->allocation_device == device)
      return RAY_CASTER_OK;

    if (task->relocate && task->relocate != &relocate_task)
      return task->relocate(task, device);

    if (device == -1)
    {
      ray_caster::task_t* rtask = ray_caster::task_create(task->n_tasks);

      // Preserve knowledge about CUDA.
      rtask->relocate = &relocate_task;

      cuda_rays_data_copy_all(task->n_tasks, rtask, task, cudaMemcpyDeviceToHost);
      cuda_rays_data_copy_all(task->next_gen.n_tasks, &rtask->next_gen, &task->next_gen, cudaMemcpyDeviceToHost);

      task_free_data(task);

      *task = *rtask;
      free(rtask);
    }
    else if (task->allocation_device == -1)
    {
      // @todo Support multi-GPU.
      ray_caster::task_t rtask = {};
      rtask.allocation_device = device;
      rtask.free_data = &cuda_task_free_data;
      rtask.grow = &grow_task;
      rtask.relocate = &relocate_task;

      int n_alloc = (task->n_tasks + 1023) & -1024;;

      rtask.n_tasks = task->n_tasks;
      rtask.next_gen.n_tasks = task->next_gen.n_tasks;
      rtask.n_reserved = n_alloc;
      rtask.n_missed = task->n_missed;

      cuda_rays_data_alloc(n_alloc, &rtask);
      cuda_rays_data_alloc(n_alloc, &rtask.next_gen);

      cuda_rays_data_copy_all(task->n_tasks, &rtask, task, cudaMemcpyHostToDevice);
      cuda_rays_data_copy_all(task->next_gen.n_tasks, &rtask.next_gen, &task->next_gen, cudaMemcpyHostToDevice);

      task_free_data(task);
      *task = rtask;
    }
    
    return RAY_CASTER_OK;
  }
}
