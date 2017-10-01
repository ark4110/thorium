// Copyright (c) 2015-2017 Contributors as noted in the AUTHORS file.
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
#include "cuda_scene_engine.cuh"

#include <helper_cuda.h>

namespace cuda_scene_engine
{
  __global__ void device_set_frame(int n_faces, const cuda_math::face_t* source, const int* face_group_idx, const cuda_math::mat33* rotation, const cuda_math::vec3* translation, cuda_math::face_t* target, cuda_math::vec3* normals)
  {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int f = index; f < n_faces; f += stride)
    {
      int group_idx = face_group_idx[f];
      for (int i = 0; i != 3; ++i)
      {
        // @todo Split it?
        target[f].points[i] = rotation[group_idx] * source[f].points[i] + translation[group_idx];
      }
      // @todo Launch separete kernel for this?
      normals[f] = normalize(face_normal(target[f]));
    }
  }

  int init(system_t* system)
  {
    int minGridSize;    
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &system->set_frame_block_size, device_set_frame, 0, 0);

    return SCENE_ENGINE_OK;
  }

  int shutdown(system_t* system)
  {
    return SCENE_ENGINE_OK;
  }

  int create_scene(system_t* system, subject::scene_t** result)
  {
    // @todo Create and later host scene?
    scene_t* scene = (scene_t*)calloc(1, sizeof(scene_t));

    scene->relocate = &cuda_relocate_scene;
    scene->free_data = &cuda_free_scene_data;
    scene->set_frame = &cuda_set_frame;

    scene->system = system;
    scene->n_current_device = -1; // After creation scene is host allocated.
    scene->bsphere = (math::sphere_t*)calloc(1, sizeof(math::sphere_t));

    scene->host_scene = (scene_t*)calloc(1, sizeof(scene_t));
    *scene->host_scene = *scene;

    *result = scene;

    return SCENE_ENGINE_OK;
  }

  int finalize_scene(system_t* system, scene_t* scene)
  {
    if (scene->n_current_device != -1)
    {
      fprintf(stderr, "Scene protocol violation\n");
      return -SCENE_ENGINE_PROTOCOL_VIOLATION;
    }

    if (int r = host_scene_finalize(scene))
      return r;
    *scene->host_scene = *scene;

    return SCENE_ENGINE_OK;
  }

  int system_set_frame(system_t* system, scene_t* scene, int n_frame)
  {
    int n_first_animated_face = scene_static_faces_count(scene->host_scene);
    int n_animated_faces = scene_animated_faces_count(scene->host_scene);

    if (!n_animated_faces)
      return SCENE_ENGINE_OK;

    if (int r = cuda_relocate_scene(scene, 0))
      return r;

    int n_grid_size = (n_animated_faces + system->set_frame_block_size - 1) / system->set_frame_block_size;

    const cuda_math::face_t* source = (const cuda_math::face_t*)&scene->base_faces[n_first_animated_face];
    const int* face_group_idx = scene->face_animation_group;
    const cuda_math::mat33* rotation = (const cuda_math::mat33*)&scene->animation->rotations[scene->animation->n_groups * n_frame];
    const cuda_math::vec3* translation = (const cuda_math::vec3*)&scene->animation->translations[scene->animation->n_groups * n_frame];
    cuda_math::face_t* target = (cuda_math::face_t*)&scene->faces[n_first_animated_face];
    cuda_math::vec3* normals = (cuda_math::vec3*)&scene->face_normals[n_first_animated_face];

    device_set_frame << < n_grid_size, system->set_frame_block_size >> > (n_animated_faces, source, face_group_idx, rotation, translation, target, normals);

    // @todo Copy from first device only
    cudaMemcpy(&scene->host_scene->faces[n_first_animated_face], &scene->faces[n_first_animated_face], n_animated_faces * sizeof(cuda_math::face_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&scene->host_scene->face_normals[n_first_animated_face], &scene->face_normals[n_first_animated_face], n_animated_faces * sizeof(float3), cudaMemcpyDeviceToHost);
    subject::build_scene_bounding_sphere(scene->host_scene);

    // @todo Pain, meet Ugly. Ugly, pain
    scene->animation->n_current_frame = n_frame;
    scene->host_scene->animation->n_current_frame = n_frame;

    return SCENE_ENGINE_OK;
  }

  const subject::system_methods_t methods =
  {
    (int(*)(subject::system_t* system))&init,
    (int(*)(subject::system_t* system))&shutdown,
    (int(*)(subject::system_t* system, subject::scene_t** scene))&create_scene,
    (int(*)(subject::system_t* system, subject::scene_t* scene))&finalize_scene,
  };

  subject::system_t* system_create()
  {
    subject::system_t* system = (subject::system_t*)calloc(1, sizeof(system_t));
    system->methods = &methods;
    return system;
  }
}
