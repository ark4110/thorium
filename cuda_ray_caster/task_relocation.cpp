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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "task_relocation.h"
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

namespace cuda_task_relocation
{
  void cuda_task_free_data(ray_caster::task_t* task)
  {
    cudaFree(task->ray);
    cudaFree(task->hit_point);
    cudaFree(task->hit_face_idx);
  }

  int grow_task(ray_caster::task_t* task, int n_rays)
  {
    int n_prev = task->n_tasks;
    if (task->n_tasks + n_rays < task->n_reserved)
    {
      task->n_tasks += n_rays;
      return n_prev;
    }

    // We grow to create new rays -> no need to preserve hit_faces_idx and hit_point.

    ray_caster::task_t* buffer_task = ray_caster::task_create(task->n_tasks);
    checkCudaErrors(cudaMemcpy(buffer_task->ray, task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(buffer_task->power, task->power, task->n_tasks * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(task->ray);
    cudaFree(task->hit_point);
    cudaFree(task->hit_face_idx);
    cudaFree(task->power);

    ray_caster::task_t rtask = { task->allocation_device, &cuda_task_free_data, &grow_task, &relocate_task };
    
    rtask.n_tasks = task->n_tasks + n_rays;
    const int align_factor = 1024 * 128;
    int n_alloc = rtask.n_tasks + (align_factor - rtask.n_tasks % align_factor) % align_factor;
    rtask.n_reserved = n_alloc;

    checkCudaErrors(cudaMalloc((void**)&rtask.ray, sizeof(math::ray_t) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&rtask.hit_point, sizeof(math::vec3) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&rtask.hit_face_idx, sizeof(int) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&rtask.power, sizeof(float) * n_alloc));

    checkCudaErrors(cudaMemcpy(rtask.ray, buffer_task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(rtask.power, buffer_task->power, task->n_tasks * sizeof(float), cudaMemcpyHostToDevice));
    task_free(buffer_task);

    if (rtask.n_reserved > 100000)
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

      checkCudaErrors(cudaMemcpy(rtask->ray, task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(rtask->hit_point, task->hit_point, task->n_tasks * sizeof(math::vec3), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(rtask->hit_face_idx, task->hit_face_idx, task->n_tasks * sizeof(int), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(rtask->power, task->power, task->n_tasks * sizeof(float), cudaMemcpyDeviceToHost));
      task_free_data(task);

      *task = *rtask;
      free(rtask);
    }
    else if (task->allocation_device == -1)
    {
      // @todo Support multi-GPU.
      ray_caster::task_t rtask = { device, &cuda_task_free_data, &grow_task, &relocate_task };
      int n_alloc = task->n_tasks + (1024 - task->n_tasks % 1024) % 1024;

      rtask.n_tasks = task->n_tasks;
      rtask.n_reserved = n_alloc;

      checkCudaErrors(cudaMalloc((void**)&rtask.ray, sizeof(math::ray_t) * n_alloc));
      checkCudaErrors(cudaMalloc((void**)&rtask.hit_point, sizeof(math::vec3) * n_alloc));
      checkCudaErrors(cudaMalloc((void**)&rtask.hit_face_idx, sizeof(int) * n_alloc));
      checkCudaErrors(cudaMalloc((void**)&rtask.power, sizeof(float) * n_alloc));

      checkCudaErrors(cudaMemcpy(rtask.ray, task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(rtask.hit_point, task->hit_point, task->n_tasks * sizeof(math::vec3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(rtask.hit_face_idx, task->hit_face_idx, task->n_tasks * sizeof(int), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(rtask.power, task->power, task->n_tasks * sizeof(float), cudaMemcpyHostToDevice));

      task_free_data(task);
      *task = rtask;
    }
    
    return RAY_CASTER_OK;
  }
}
