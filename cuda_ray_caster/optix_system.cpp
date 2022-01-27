// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
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

#include "optix_system.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory>
#include <optix_prime/optix_prime.h>
#include "optix_hit_converter.h"
#include "task_relocation.h"

namespace optix_ray_caster
{
  struct system_t : ray_caster::system_t
  {
    bool use_cpu;
    subject::scene_t* scene;
    int last_update_frame;

    RTPcontexttype context_type;
    
    RTPcontext context;
    RTPbufferdesc vertices;
    RTPmodel model;

    optix_hit_converter::state_t* converter_state;
  };

  const char* error_to_text(int r)
  {
    switch (r)
    {
    case RTP_ERROR_OBJECT_CREATION_FAILED:
      return "RTP_ERROR_OBJECT_CREATION_FAILED";

    case RTP_ERROR_INVALID_VALUE:
      return "RTP_ERROR_INVALID_VALUE";

    case RTP_ERROR_MEMORY_ALLOCATION_FAILED:
      return "RTP_ERROR_MEMORY_ALLOCATION_FAILED";

    default:
      return "unknown error";
    }
  }

  int init(system_t* system)
  {
    system->scene = 0;
    system->context = {};

    system->context_type = RTP_CONTEXT_TYPE_CUDA;
    int r = 0;
    if (system->use_cpu)
    {
      system->context_type = RTP_CONTEXT_TYPE_CPU;
      if (rtpContextCreate(RTP_CONTEXT_TYPE_CPU, &system->context) != RTP_SUCCESS)
        return -RAY_CASTER_ERROR;
    }
    else if ((r = rtpContextCreate(system->context_type, &system->context)) == RTP_SUCCESS)
    {
      unsigned devices[] = { 0 };
      if (rtpContextSetCudaDeviceNumbers(system->context, 1, devices) != RTP_SUCCESS)
      {
        rtpContextDestroy(system->context);
        return -RAY_CASTER_ERROR;
      }
    }
    else
    {
      system->context_type = RTP_CONTEXT_TYPE_CPU;
      fprintf(stderr, "Optix ray caster fallback to CPU! Reason %s \n", error_to_text(r));
      if (rtpContextCreate(RTP_CONTEXT_TYPE_CPU, &system->context) != RTP_SUCCESS)
        return -RAY_CASTER_ERROR;
    }

    system->vertices = {};
    system->model = {};

    system->converter_state = 0;

    return RAY_CASTER_OK;
  }

  int shutdown(system_t* system)
  {
    if (system->context)
      rtpContextDestroy(system->context);
    system->context = 0;

    state_free(system->converter_state);
    system->converter_state = 0;

    return RAY_CASTER_OK;
  }

  void system_release_scene_data(system_t* system)
  {
    rtpBufferDescDestroy(system->vertices);
    system->vertices = {};
    
    rtpModelDestroy(system->model);
    system->model = {};

    system->scene = 0;
  }

  int set_scene(system_t* system, subject::scene_t* scene)
  {
    if (system->scene)
      system_release_scene_data(system);

    system->scene = scene;
    system->last_update_frame = -1;

    if (scene->n_faces == 0)
      return RAY_CASTER_OK;

    int r;
    bool relocation_supported = true;
    if ((r = scene_relocate(scene, 0)) != SCENE_ENGINE_OK)
    {
      if (r == -SCENE_ENGINE_NOT_SUPPORTED)
        relocation_supported = false;
      else
        return r;
    }

    if (rtpBufferDescCreate(system->context,
      RTP_BUFFER_FORMAT_VERTEX_FLOAT3,
      relocation_supported ? RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST, scene->faces, &system->vertices) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetRange(system->vertices, 0, scene->n_faces * 3) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetStride(system->vertices, 0) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    if (rtpModelCreate(system->context, &system->model) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    if (false)
    {
      if (rtpModelSetTriangles(system->model, 0, system->vertices) != RTP_SUCCESS)
        return -RAY_CASTER_ERROR;
      if (rtpModelUpdate(system->model, RTP_MODEL_HINT_NONE) != RTP_SUCCESS)
        return -RAY_CASTER_ERROR;
      if (rtpModelFinish(system->model) != RTP_SUCCESS)
        return -RAY_CASTER_ERROR;
    }
    
    bool force_cpu = system->context_type == RTP_CONTEXT_TYPE_CPU;
    if (system->converter_state)
      state_free(system->converter_state);
    system->converter_state = optix_hit_converter::state_init(scene, force_cpu);

    if ((r = scene_relocate(scene, -1)))
      return r;

    return RAY_CASTER_OK;
  }

  int update(system_t* system)
  {
    if (!system->scene || system->scene->n_faces == 0)
      return -RAY_CASTER_ERROR;

    if (system->scene->animation)
    {
      // @todo Init with frame number 0.
      if (system->scene->animation->n_current_frame == system->last_update_frame && system->scene->animation->n_current_frame != -1)
        return RAY_CASTER_OK;
      else
        system->last_update_frame = system->scene->animation->n_current_frame;
    }

    int r;
    if ((r = rtpModelSetTriangles(system->model, 0, system->vertices)) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if ((r = rtpModelUpdate(system->model, RTP_MODEL_HINT_ASYNC)) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    return RAY_CASTER_OK;
  }

  int cast(system_t* system, ray_caster::task_t* task)
  {
    if (task->n_tasks == 0)
      return RAY_CASTER_OK;

    int r;
    if ((r = rtpModelFinish(system->model)) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    
    int c = 0;
    RTPbufferdesc rays;
    if (rtpBufferDescCreate(system->context,
      RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION,
      (task->allocation_device == -1 ? RTP_BUFFER_TYPE_HOST : RTP_BUFFER_TYPE_CUDA_LINEAR), task->ray, &rays) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetRange(rays, 0, task->n_tasks) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    optix_hit_converter::state_t* converter = system->converter_state;
    if ((c = state_prepare(converter, task)) < 0)
    {
      rtpBufferDescDestroy(rays);
      return c;
    }

    RTPbufferdesc hits;
    if (rtpBufferDescCreate(system->context,
      RTP_BUFFER_FORMAT_HIT_T_TRIID,
      RTPbuffertype(converter->optix_buffer_type),
      converter->hits, &hits) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetRange(hits, 0, task->n_tasks) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    RTPquery query;
    if (rtpQueryCreate(system->model, RTP_QUERY_TYPE_CLOSEST, &query) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpQuerySetRays(query, rays) != RTP_SUCCESS)
    {
      rtpQueryDestroy(query);
      return -RAY_CASTER_ERROR;
    }
    if (rtpQuerySetHits(query, hits) != RTP_SUCCESS)
    {
      rtpQueryDestroy(query);
      return -RAY_CASTER_ERROR;
    }
    if ((r = rtpQueryExecute(query, 0)) != RTP_SUCCESS)
    {
      rtpQueryDestroy(query);
      return -RAY_CASTER_ERROR;
    }
    rtpQueryDestroy(query);

    r = convert(converter, task);

    rtpBufferDescDestroy(hits);
    rtpBufferDescDestroy(rays);
   
    return r;
  }

  int relocate(system_t* system, ray_caster::task_t* task, int device)
  {
    if (device == 0)
      device = system->context_type == RTP_CONTEXT_TYPE_CUDA ? 1 : -1;
    return cuda_task_relocation::relocate_task(task, device);
  }

  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&update,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task, int device))&relocate,
  };

  ray_caster::system_t* system_create(bool use_cpu)
  {
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    s->use_cpu = use_cpu;
    return s;
  }
}
