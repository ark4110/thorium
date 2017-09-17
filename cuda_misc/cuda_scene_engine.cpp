// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Thorium is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.


#include "cuda_scene_engine.h"
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

namespace cuda_scene_engine
{
  int relocate_cuda_scene(scene_t* scene, int device)
  {
    if (scene->n_current_device == device)
      return SCENE_ENGINE_OK;

    if (device == 0 && scene->n_current_device == 1)
      return SCENE_ENGINE_OK;

    if (device == -1)
    {
      *static_cast<subject::scene_t*>(scene) = *scene->host_scene;
      scene->n_current_device = -1;
      return SCENE_ENGINE_OK;
    }

    if (device == 0)
      device = 1;

    int device_idx = device - 1;
    if (scene->n_devices > device_idx && scene->device_scenes && scene->device_scenes[device_idx])
    {
      *static_cast<subject::scene_t*>(scene) = *scene->device_scenes[device_idx];
      scene->n_current_device = device;
      return SCENE_ENGINE_OK;
    }

    if (scene->n_devices < device)
    {
      scene->device_scenes = (subject::scene_t**)realloc(scene->device_scenes, (1 + device) * sizeof(subject::scene_t*));
      for (int d = scene->n_devices; d != device; ++d)
        scene->device_scenes[d] = 0;
      scene->n_devices = device;
    }

    scene->device_scenes[device_idx] = (subject::scene_t*)malloc(sizeof(subject::scene_t));
    *scene->device_scenes[device_idx] = *scene->host_scene;

    // @todo Set CUDA device here.
    subject::scene_t* source = scene->host_scene; // host is source
    subject::scene_t* target = scene->device_scenes[device_idx];

    checkCudaErrors(cudaMalloc((void**)&target->faces, sizeof(subject::face_t) * source->n_faces));
    checkCudaErrors(cudaMemcpy(target->faces, source->faces, sizeof(subject::face_t) * source->n_faces, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&target->meshes, sizeof(subject::mesh_t) * source->n_meshes));
    checkCudaErrors(cudaMemcpy(target->meshes, source->meshes, sizeof(subject::mesh_t) * source->n_meshes, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&target->materials, sizeof(subject::material_t) * source->n_materials));
    checkCudaErrors(cudaMemcpy(target->materials, source->materials, sizeof(subject::material_t) * source->n_materials, cudaMemcpyHostToDevice));

    if (source->animation)
    {
      target->animation = (subject::animation_t*)calloc(1, sizeof(subject::animation_t));
      *target->animation = *source->animation;
      checkCudaErrors(cudaMalloc((void**)&target->animation->groups, sizeof(subject::face_group_t) * source->animation->n_groups));
      checkCudaErrors(cudaMemcpy(target->animation->groups, source->animation->groups, sizeof(subject::face_group_t) * source->animation->n_groups, cudaMemcpyHostToDevice));

      checkCudaErrors(cudaMalloc((void**)&target->animation->translations, sizeof(math::vec3) * source->animation->n_groups * source->animation->n_frames));
      checkCudaErrors(cudaMemcpy(target->animation->translations, source->animation->translations, sizeof(math::vec3) * source->animation->n_groups * source->animation->n_frames, cudaMemcpyHostToDevice));

      checkCudaErrors(cudaMalloc((void**)&target->animation->rotations, sizeof(math::mat33) * source->animation->n_groups * source->animation->n_frames));
      checkCudaErrors(cudaMemcpy(target->animation->rotations, source->animation->rotations, sizeof(math::mat33) * source->animation->n_groups * source->animation->n_frames, cudaMemcpyHostToDevice));

      int n_first_animated_face = source->animation->groups[0].n_offset;
      int n_animated_faces = source->n_faces - n_first_animated_face;
      checkCudaErrors(cudaMalloc((void**)&target->face_animation_group, sizeof(int) * n_animated_faces));
      checkCudaErrors(cudaMemcpy(target->face_animation_group, source->face_animation_group, sizeof(int) * n_animated_faces, cudaMemcpyHostToDevice));
    }

    checkCudaErrors(cudaMalloc((void**)&target->face_areas, sizeof(float) * source->n_faces));
    checkCudaErrors(cudaMemcpy(target->face_areas, source->face_areas, sizeof(float) * source->n_faces, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&target->face_normals, sizeof(math::vec3) * source->n_faces));
    checkCudaErrors(cudaMemcpy(target->face_normals, source->face_normals, sizeof(math::vec3) * source->n_faces, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&target->face_to_mesh, sizeof(int) * source->n_faces));
    checkCudaErrors(cudaMemcpy(target->face_to_mesh, source->face_to_mesh, sizeof(int) * source->n_faces, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&target->mesh_material, sizeof(int) * source->n_meshes));
    checkCudaErrors(cudaMemcpy(target->mesh_material, source->mesh_material, sizeof(int) * source->n_meshes, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&target->base_faces, sizeof(subject::face_t) * source->n_faces));
    checkCudaErrors(cudaMemcpy(target->base_faces, source->base_faces, sizeof(subject::face_t) * source->n_faces, cudaMemcpyHostToDevice));

    *static_cast<subject::scene_t*>(scene) = *target;
    scene->n_current_device = device;

    return SCENE_ENGINE_OK;
  }

  int cuda_relocate_scene(subject::scene_t* scene, int device)
  {
    return relocate_cuda_scene(static_cast<scene_t*>(scene), device);
  }

  void free_cuda_scene_data(scene_t* scene)
  {
    for (int d = 0; d != scene->n_devices; ++d)
    {
      subject::scene_t* target = scene->device_scenes[d];
      if (!target)
        continue;

      // @todo Set CUDA device here.
      cudaFree(target->faces);
      cudaFree(target->meshes);
      cudaFree(target->materials);
      if (target->animation)
      {
        cudaFree(target->animation->groups);
        cudaFree(target->animation->translations);
        cudaFree(target->animation->rotations);
      }
      cudaFree(target->face_to_mesh);
      cudaFree(target->face_areas);
      cudaFree(target->face_normals);
      cudaFree(target->mesh_material);
      cudaFree(target->base_faces);

      free(target);
    }
    free(scene->device_scenes);

    subject::host_scene_free_data(scene->host_scene);
  }

  void cuda_free_scene_data(subject::scene_t* scene)
  {
    free_cuda_scene_data(static_cast<scene_t*>(scene));
  }

  int cuda_set_frame(subject::scene_t* scene, int n_frame)
  { 
    scene_t* cuda_scene = static_cast<scene_t*>(scene);

    // Forward a request to the system that knowns about hardware.
    return system_set_frame(cuda_scene->system, cuda_scene, n_frame);
  }  
}
