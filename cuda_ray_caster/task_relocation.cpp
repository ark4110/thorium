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
    ray_caster::task_t rtask = { task->allocation_device, &cuda_task_free_data, &grow_task, &relocate_task };
    
    rtask.n_tasks = task->n_tasks + n_rays;
    int n_alloc = rtask.n_tasks + (1024 - rtask.n_tasks % 1024) % 1024;
    rtask.n_reserved = n_alloc;

    checkCudaErrors(cudaMalloc((void**)&rtask.ray, sizeof(math::ray_t) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&rtask.hit_point, sizeof(math::vec3) * n_alloc));
    checkCudaErrors(cudaMalloc((void**)&rtask.hit_face_idx, sizeof(int) * n_alloc));

    checkCudaErrors(cudaMemcpy(rtask.ray, task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(rtask.hit_point, task->hit_point, task->n_tasks * sizeof(math::vec3), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(rtask.hit_face_idx, task->hit_face_idx, task->n_tasks * sizeof(int), cudaMemcpyDeviceToDevice));

    cudaFree(task->ray);
    cudaFree(task->hit_point);
    cudaFree(task->hit_face_idx);

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

      checkCudaErrors(cudaMemcpy(rtask->ray, task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(rtask->hit_point, task->hit_point, task->n_tasks * sizeof(math::vec3), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(rtask->hit_face_idx, task->hit_face_idx, task->n_tasks * sizeof(int), cudaMemcpyDeviceToHost));
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

      checkCudaErrors(cudaMemcpy(rtask.ray, task->ray, task->n_tasks * sizeof(math::ray_t), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(rtask.hit_point, task->hit_point, task->n_tasks * sizeof(math::vec3), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(rtask.hit_face_idx, task->hit_face_idx, task->n_tasks * sizeof(int), cudaMemcpyHostToDevice));

      task_free_data(task);
      *task = rtask;
    }
    
    return RAY_CASTER_OK;
  }
}
