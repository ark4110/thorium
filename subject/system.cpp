// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

/**
* This module contains basic types for scene definition and manipulation.
*/

#include "system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include "../cuda_misc/cuda_scene_engine.h"
#include <stdlib.h>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <vector>

namespace subject
{
  void host_scene_free_data(scene_t* scene)
  {
    free(scene->faces);
    for (int m = 0; m != scene->n_meshes; ++m)
      free(scene->meshes[m].name);
    free(scene->meshes);
    free(scene->materials);

    free(scene->face_areas);
    free(scene->face_normals);
    free(scene->face_to_mesh);
    free(scene->face_animation_group);
    free(scene->mesh_material);
    free(scene->mesh_areas);

    free(scene->base_faces);

    animation_free(scene->animation);
  }
  
  void scene_free(scene_t* scene)
  {
    if (!scene)
      return;
    if (scene->free_data)
      scene->free_data(scene);
    free(scene);
  }

  scene_t* scene_create()
  {
    scene_t* s = (scene_t*)calloc(1, sizeof(scene_t));
    s->free_data = &host_scene_free_data;
    s->relocate = &host_scene_relocate;
    s->set_frame = &host_scene_set_frame;

    return s;
  }

  int scene_set_frame(scene_t* scene, int n_frame)
  {
    if (!scene->animation)
      return SCENE_ENGINE_OK;

    animation_t* anim = scene->animation;
    if (anim->n_current_frame == n_frame)
      return SCENE_ENGINE_OK;
    int r = scene->set_frame(scene, n_frame);
    anim->n_current_frame = n_frame;
    return r;
  }

  int scene_relocate(scene_t* scene, int device)
  {
    return scene->relocate(scene, device);
  }

  animation_t* animation_create(int n_groups, int n_frames)
  {
    animation_t* anim = (animation_t*)calloc(1, sizeof(animation_t));
    anim->n_groups = n_groups;
    anim->n_frames = n_frames;
    anim->n_current_frame = -1;
    anim->groups = (face_group_t*)malloc(n_groups * sizeof(face_group_t));
    anim->translations = (math::vec3*)malloc(n_groups * n_frames * sizeof(math::vec3));
    anim->rotations = (math::mat33*)malloc(n_groups * n_frames * sizeof(math::mat33));
    return anim;
  }

  void animation_free(animation_t* anim)
  {
    if (!anim)
      return;
    free(anim->groups);
    free(anim->translations);
    free(anim->rotations);
    free(anim);
  }

  task_t* task_create()
  {
    task_t* t = (task_t*)calloc(1, sizeof(task_t));
    *t = { 0, // orbit
      1000000, 1000000, // rays_counts
      1.f, // frame_time_scale
      60.f // duration
    };

    return t;
  }

  void task_free(task_t* task)
  {
    if (!task)
      return;
    
    tle_orbit_free(task->tle_orbit);

    free(task);
  }

  int host_scene_finalize(scene_t* scene)
  { 
    if (int r = host_scene_sort_groups(scene))
      return r;

    build_faces_areas(scene, &scene->face_areas);
    build_face_normals(scene, &scene->face_normals);
    build_face_to_mesh_index(scene, &scene->face_to_mesh);    
    build_face_animation_group_index(scene, &scene->face_animation_group);
    build_mesh_material_index(scene, &scene->mesh_material);
    build_meshes_areas(scene, &scene->mesh_areas);

    if (scene->animation && scene->animation->groups)
    {
      scene->base_faces = (face_t*)malloc(scene->n_faces * sizeof(face_t));
      memcpy(scene->base_faces, scene->faces, scene->n_faces * sizeof(face_t));
    }

    return SCENE_ENGINE_OK;
  }

  int host_scene_set_frame(scene_t* scene, int n_frame)
  {
    if (!scene->animation)
      return SCENE_ENGINE_OK;

    if (scene->animation->n_current_frame == n_frame)
      return SCENE_ENGINE_OK;

    const animation_t* anim = scene->animation;
    if (n_frame >= anim->n_frames)
      return -SCENE_ENGINE_FRAME_OUT_OF_RANGE;

    math::vec3* face_normals = scene->face_normals;
    const math::vec3* translation = &anim->translations[n_frame * anim->n_groups];
    const math::mat33* rotation = &anim->rotations[n_frame * anim->n_groups];
    face_t* source = scene->base_faces;
    face_t* target = scene->faces;
    for (int g = 0; g != anim->n_groups; ++g)
    {
      const face_group_t& group = anim->groups[g];
      for (int f = group.n_offset; f != group.n_offset + group.n_length; ++f)
      {
        for (int i = 0; i != 3; ++i)
          target[f].points[i] = rotation[g] * source[f].points[i] + translation[g];
        face_normals[f] = normalize(triangle_normal(target[f]));
      }
    }

    return SCENE_ENGINE_OK;
  }

  template< typename order_iterator, typename value_iterator >
  void reorder(order_iterator order_iter, order_iterator order_end, value_iterator values)
  {
    typedef typename std::iterator_traits< value_iterator >::value_type value_t;
    std::vector<value_t> ordered;
    ordered.reserve(order_end - order_iter);

    for (; order_iter != order_end; ++order_iter)
      ordered.push_back(values[*order_iter]);

    int idx = 0;
    for (auto v : ordered)
      values[idx++] = v;
  }

  int host_scene_sort_groups(scene_t* scene)
  {
    if (!scene->animation || scene->animation->n_groups == 0)
      return SCENE_ENGINE_OK;

    animation_t* animation = scene->animation;
    
    std::vector<int> order(animation->n_groups);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> correct_order = order;
    std::sort(order.begin(), order.end(), [animation](int l, int r) { return animation->groups[l].n_offset < animation->groups[l].n_offset; });

    if (order != correct_order)
    {
      reorder(order.begin(), order.end(), &animation->groups);
      for (int f = 0; f != animation->n_frames; ++f)
      {
        reorder(order.begin(), order.end(), &animation->rotations[f * animation->n_groups]);
        reorder(order.begin(), order.end(), &animation->translations[f * animation->n_groups]);
      }
    }

    int n_first_animated = animation->groups[0].n_offset;
    int n_total_animated = std::accumulate(animation->groups, animation->groups + animation->n_groups, 0, [](int s, const face_group_t& g) { return s + g.n_length; });

    // sanity check
    if (scene->n_faces != n_first_animated + n_total_animated)
      return -SCENE_ENGINE_ANIMATION_GROUPS_BROKEN;

    // groups overlap check
    for (int i = 1; i != animation->n_groups; ++i)
    {
      const face_group_t& prev = animation->groups[i - 1];
      const face_group_t& current = animation->groups[i];
      if (prev.n_offset + prev.n_length > current.n_offset)
        return -SCENE_ENGINE_ANIMATION_GROUPS_BROKEN;
    }

    struct face_group_less_t
    {
      bool operator()(const face_group_t& l, const face_group_t& r) const
      {
        return l.n_offset < r.n_offset;
      }

      bool operator()(const face_group_t& l, int r) const
      {
        return l.n_offset < r;
      }

      bool operator()(int l, const face_group_t& r) const
      {
        return l < r.n_offset;
      }
    };

    // mesh groups boundary overlap check
    for (int m = 0; m != scene->n_meshes; ++m)
    {
      const mesh_t& mesh = scene->meshes[m];
      if (mesh.first_idx < n_first_animated && mesh.first_idx + mesh.n_faces > n_first_animated)
        return -SCENE_ENGINE_ANIMATION_GROUPS_BROKEN;
      auto found = std::lower_bound(animation->groups, animation->groups + animation->n_groups, mesh.first_idx, face_group_less_t());
      if (found != animation->groups + animation->n_groups)
      {
        if (mesh.first_idx + mesh.n_faces > found->n_offset + found->n_length)
          return -SCENE_ENGINE_ANIMATION_GROUPS_BROKEN;
      }
    }

    return SCENE_ENGINE_OK;
  }

  int host_scene_relocate(scene_t* scene, int device)
  {
    if (device == -1)
      return SCENE_ENGINE_OK;
    else
      return -SCENE_ENGINE_NOT_SUPPORTED;
  }

  int scene_static_faces_count(const scene_t* scene)
  {
    return scene->animation ? scene->animation->n_static_faces : scene->n_faces;
  }

  int scene_animated_faces_count(const scene_t* scene)
  {
    return scene->animation ? scene->animation->n_dynamic_faces : 0;
  }

  namespace subject_engine_host
  {
    struct system_t : subject::system_t
    {
    };

    int init(system_t* system)
    {
      return SCENE_ENGINE_OK;
    }

    int shutdown(system_t* system)
    {
      return SCENE_ENGINE_OK;
    }

    int create_scene(system_t* system, scene_t** target)
    {
      *target = scene_create();
      return SCENE_ENGINE_OK;
    }

    int finalize_scene(system_t* system, scene_t* scene)
    {
      return host_scene_finalize(scene);
    }

    /// @brief Creates virtual methods table from local methods.
    const subject::system_methods_t methods =
    {
      (int(*)(subject::system_t* system))&init,
      (int(*)(subject::system_t* system))&shutdown,
      (int(*)(subject::system_t* system, subject::scene_t** scene))&create_scene,
      (int(*)(subject::system_t* system, subject::scene_t* scene))&finalize_scene,
    };

    /// @brief Creates base system for ray caster.
    subject::system_t* system_create()
    {
      system_t* s = (system_t*)calloc(1, sizeof(system_t));
      s->methods = &methods;
      return s;
    }
  }

  system_t* system_create(int type)
  {
    system_t* system = 0;
    switch (type)
    {
    case SCENE_ENGINE_HOST:
      system = subject_engine_host::system_create();
      break;

    case SCENE_ENGINE_CUDA:
      system = cuda_scene_engine::system_create();
      break;

    default:
      fprintf(stderr, "Unknown engine system type %d.\n", type);
    }

    if (system)
      system_init(system);

    return system;
  }

  void system_free(system_t* system)
  {
    if (system)
      system_shutdown(system);
    free(system);
  }

  int system_init(system_t* system)
  {
    return system->methods->init(system);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_create_scene(system_t* system, scene_t** scene)
  {
    return system->methods->create_scene(system, scene);
  }

  int system_finalize_scene(system_t* system, scene_t* scene)
  {
    return system->methods->finalize_scene(system, scene);
  }

  float mesh_area(const scene_t* scene, const mesh_t& mesh)
  {
    double area = 0;
    for (int f = 0; f != mesh.n_faces; ++f)
      area += scene->face_areas[mesh.first_idx + f];
    return (float)area;
  }

  float mesh_area(const scene_t* scene, int mesh_idx)
  {
    return mesh_area(scene, scene->meshes[mesh_idx]);
  }

  float build_meshes_areas(const scene_t* scene, float** areas)
  {
    *areas = (float*)realloc(*areas, sizeof(float) * scene->n_meshes);

    float total_area = 0;
    const int n_meshes = scene->n_meshes;
    for (int m = 0; m != n_meshes; ++m)
    {
      float area = mesh_area(scene, m);
      total_area += area;
      (*areas)[m] = area;
    }

    return total_area;
  }

  void build_faces_areas(const scene_t* scene, float** areas)
  {
    *areas = (float*)realloc(*areas, sizeof(float) * scene->n_faces);

    const int n_faces = scene->n_faces;
    for (int f = 0; f != n_faces; ++f)
    {
      float area = math::triangle_area(scene->faces[f]);
      (*areas)[f] = area;
    }
  }

  void build_face_to_mesh_index(int n_faces, int n_meshes, const mesh_t* meshes, int** index)
  {
    free(*index);
    *index = (int*)malloc(n_faces * sizeof(int));

    // fill face-to-mesh inverted index for every mesh
    for (int m = 0; m != n_meshes; ++m)
    {
      const mesh_t& mesh = meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        (*index)[mesh.first_idx + f] = m;
      }
    }
  }

  void build_face_to_mesh_index(const scene_t* scene, int** index)
  {
    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, index);
  }

  void build_face_normals(const scene_t* scene, math::vec3** normals)
  {
    if (!*normals)
      *normals = (math::vec3*)malloc(scene->n_faces * sizeof(math::vec3));
    
    for (int f = 0; f != scene->n_faces; ++f)
      (*normals)[f] = normalize(triangle_normal(scene->faces[f]));
  }

  void build_mesh_material_index(const scene_t* scene, int** index)
  {
    free(*index);
    *index = (int*)malloc(scene->n_meshes * sizeof(int));
    for (int m = 0; m != scene->n_meshes; ++m)
    {
      (*index)[m] = scene->meshes[m].material_idx;
    }
  }

  void build_face_animation_group_index(const scene_t* scene, int** index)
  {
    if (!scene->animation)
      return;
    const int n_first_animated_face = scene->animation->groups[0].n_offset;
    const int n_animated_faces = scene->n_faces - n_first_animated_face;

    if (!*index)
      *index = (int*)malloc(n_animated_faces * sizeof(int));

    const animation_t* animation = scene->animation;
    for (int g = 0; g != animation->n_groups; ++g)
    {
      const face_group_t& group = animation->groups[g];
      for (int i = 0; i != group.n_length; ++i)
        (*index)[group.n_offset + i - n_first_animated_face] = g;
    }
  }

  const material_t& mesh_material(const scene_t* scene, int mesh_idx)
  {
    return scene->materials[scene->meshes[mesh_idx].material_idx];
  }
}
