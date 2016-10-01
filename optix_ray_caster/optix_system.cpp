// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
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

#include "optix_system.h"
#include <stdlib.h>
#include <stdio.h>
#include <memory>
#include <optix_prime/optix_prime.h>
#include "../math/operations.h"
#include "../math/triangle.h"

namespace optix_ray_caster
{
  struct system_t : ray_caster::system_t
  {
    ray_caster::scene_t* scene;
    
    RTPcontext context;
    RTPbufferdesc vertices;
    RTPmodel model;
  };

  int init(system_t* system)
  {
    system->scene = 0;

    system->context = {};
    if (rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &system->context) == RTP_SUCCESS)
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
      fprintf(stderr, "Optix ray caster fallback to CPU!\n");
      if (rtpContextCreate(RTP_CONTEXT_TYPE_CPU, &system->context) != RTP_SUCCESS)
        return -RAY_CASTER_ERROR;
    }
     

    system->vertices = {};
    system->model = {};

    return RAY_CASTER_OK;
  }

  int shutdown(system_t* system)
  {
    if (system->context)
      rtpContextDestroy(system->context);
    system->context = 0;

    return RAY_CASTER_OK;
  }

  int set_scene(system_t* system, ray_caster::scene_t* scene)
  {
    system->scene = scene;

    if (scene->n_faces == 0)
      return RAY_CASTER_OK;

    if (rtpBufferDescCreate(system->context,
      RTP_BUFFER_FORMAT_VERTEX_FLOAT3,
      RTP_BUFFER_TYPE_HOST, scene->faces, &system->vertices) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetRange(system->vertices, 0, scene->n_faces * 3) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetStride(system->vertices, 0) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    if (rtpModelCreate(system->context, &system->model) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpModelSetTriangles(system->model, 0, system->vertices) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpModelUpdate(system->model, RTP_MODEL_HINT_NONE) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpModelFinish(system->model) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    return RAY_CASTER_OK;
  }

  int prepare(system_t* system)
  {
    if (!system->scene || system->scene->n_faces == 0)
      return -RAY_CASTER_ERROR;
    return RAY_CASTER_OK;
  }

  int cast(system_t* system, ray_caster::task_t* task)
  {
    if (task->n_tasks == 0)
      return RAY_CASTER_OK;

    math::ray_t* directed_rays = (math::ray_t*)malloc(task->n_tasks * sizeof(math::ray_t));
    memcpy(directed_rays, task->ray, task->n_tasks * sizeof(math::ray_t));
    for (int r = 0; r != task->n_tasks; ++r)
      directed_rays[r].direction -= directed_rays[r].origin;
    
    int c = 0;
    RTPbufferdesc rays;
    if (rtpBufferDescCreate(system->context,
      RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION,
      RTP_BUFFER_TYPE_HOST, directed_rays, &rays) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetRange(rays, 0, task->n_tasks) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    struct optix_hit
    {
      float distance;
      int triangle_id;
      float u;
      float v;
    };    

    optix_hit* hits_buffer = (optix_hit*)malloc(task->n_tasks * sizeof(optix_hit));

    RTPbufferdesc hits;
    if (rtpBufferDescCreate(system->context,
      RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V,
      RTP_BUFFER_TYPE_HOST, hits_buffer, &hits) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;
    if (rtpBufferDescSetRange(hits, 0, task->n_tasks) != RTP_SUCCESS)
      return -RAY_CASTER_ERROR;

    RTPquery query;
    rtpQueryCreate(system->model, RTP_QUERY_TYPE_CLOSEST, &query);
    rtpQuerySetRays(query, rays);
    rtpQuerySetHits(query, hits);
    rtpQueryExecute(query, 0);
    rtpQueryDestroy(query);

    for (int r = 0; r != task->n_tasks; ++r)
    {
      if (hits_buffer[r].distance < 0)
        task->hit_face[r] = 0;
      else
      {        
        const optix_hit& hit = hits_buffer[r];
        task->hit_face[r] = system->scene->faces + hit.triangle_id;
        task->hit_point[r] = triangle_uv_to_point(system->scene->faces[hit.triangle_id], hit.u, hit.v);
      }
    }

    rtpBufferDescDestroy(hits);
    rtpBufferDescDestroy(rays);

    free(hits_buffer);
    free(directed_rays);

    return RAY_CASTER_OK;
  }

  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&prepare,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  ray_caster::system_t* system_create()
  {
    system_t* s = (system_t*)malloc(sizeof(system_t));
    s->methods = &methods;
    return s;
  }
}
