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
// along with thorium.  If not, see <http://www.gnu.org/licenses/>.

/**
* This module contains basic types for scene definition and manipulation.
* @todo Provide correct file name instead of system.h
*/

#pragma once

#include "../math/types.h"
#include "../ballistics/tle_orbit.h"

namespace subject
{
  typedef double mesh_emission_t;

  /// @brief Face (polygon) type
  typedef math::triangle_t face_t;
  
  struct shell_properties_t
  {
    float density; ///< kg/m^3
    float heat_capacity; ///< dQ/dT J/(kg*K)
    float thermal_conductivity; ///< W/(m*K)
    float thickness; ///< m, <= 0 for solid
  };

  shell_properties_t default_shell_properties();

  /// @brief Optical material properties - i.e. what happens with incoming light
  struct optical_properties_t
  {
    float specular_reflectance;
    float diffuse_reflectance;
    float absorbance; ///< Коэффициент поглощения (rus.)
    float transmittance; ///< The ratio of the light energy falling on a body to that transmitted through it. Коэффициент пропускания (rus.)
  };

  /// @brief Properties of material that influence outgoing radiation
  struct emission_properties_t
  {
    float emissivity; ///< emissivity coefficient - effectiveness in emitting energy as thermal radiation
  };

  struct radiance_properties_t
  {
    optical_properties_t optical;
    emission_properties_t emission;
  };

  radiance_properties_t black_material();

  struct material_t
  {
    shell_properties_t shell;
    radiance_properties_t front;
    radiance_properties_t rear;
    char name[128];
  };

  /// @brief Black body with major parameters equal to 1.f
  material_t black_body();

  /// @brief Mesh type (group of polygons - as offset in whole scene polygons).
  struct mesh_t
  {
    int first_idx;
    int n_faces;
    int material_idx;
    char* name;
  };

  inline mesh_t make_mesh(int first_idx, int n_faces, int material_idx)
  {
    return{ first_idx, n_faces, material_idx };
  }

  /**
    @detail Must be sorted by offset.
    @detail Not animated geometry must be placed first, i.e. assert(first_animated_face_idx == animation->groups[0].n_offset).
    @todo Ensure that data is properly prepared in system_finalize_scene().
  */
  struct face_group_t
  {
    int n_offset;
    int n_length;
  };

  struct animation_t
  {
    int n_groups;
    int n_frames;
    int n_static_faces;
    int n_dynamic_faces;
    int n_current_frame;
    float frame_time_step;
    face_group_t* groups;
    math::vec3 * translations; // n_frames x n_groups
    math::mat33* rotations; // n_frames x n_groups
  };

  /**
    @brief Face graph walker callback.
    @param current_idx Zero-based face index.
    @param leaf_idx Zero-based face adjacent to current face. -1 if there are no adjacent faces to current face.
    @param vertex_mapping Mapping of adjacent vertices (see math::triangle_find_adjacent_vertices()) . Current face is 'left'.
    @param have_more Signal if there are more adjacent faces.
    @param param Data passed to initial walk() function.
    @return 0 to continue, >0 to stop iteration, <0 for error.
  */
  typedef int(*face_graph_walker)(int current_idx, int leaf_idx, int vertex_mapping, bool have_more, void* param);

  /**
    @brief Walk face graph.
    @note Algorithm complexity is ~N^2, there N is faces count.
  */
  int face_walk_graph_n2c(const face_t* faces, int n_faces, face_graph_walker walker, void* param);

  struct face_graph_index_t
  {
    int(*walk_func)(const face_graph_index_t* system, face_graph_walker walker, void* param);
    void(*free_data)(face_graph_index_t* system);
  };

  /**
    @brief Scene representation.
    @note Must be binary compatible with ray caster scene.
  */
  struct scene_t
  {
    int n_faces; ///< Total number of polygons.
    face_t* faces; ///< Polygons array.    
    int n_meshes; ///< Number of meshes.
    mesh_t* meshes; ///< Meshes array.    
    int n_materials; ///< Total number of materials.
    material_t* materials; ///< Materials array.
    animation_t* animation;

    void(*free_data)(scene_t*);
    int(*relocate)(scene_t*, int device);
    int(*set_frame)(scene_t*, int n_frame);

    float* face_areas;
    math::vec3* face_normals;    
    int* face_to_mesh; ///< Face to its mesh index.
    int* face_animation_group; ///< Index of face animation group.
    int* blind_faces;
    face_graph_index_t* face_graph;
    float* mesh_areas;
    int* mesh_material; ///< Mesh material index without indirection.
    math::sphere_t* bsphere; ///< Scene bounding sphere. @note Not relocated.

    face_t* base_faces; ///< Faces without transformation applied.
  };

  /// @brief Simulation task parameters.
  struct task_t
  {
    ballistics::tle_orbit_t* tle_orbit;

    int n_emission_rays;
    int n_solar_rays;
    float frame_time_scale; ///< Seconds per frame.
    float duration;
  };

#define SCENE_ENGINE_HOST 0
#define SCENE_ENGINE_CUDA 1

  struct system_t
  {
    const struct system_methods_t* methods;
  };

#define SCENE_ENGINE_OK    0
#define SCENE_ENGINE_ERROR 600
#define SCENE_ENGINE_NOT_SUPPORTED 601
#define SCENE_ENGINE_FRAME_OUT_OF_RANGE 602
#define SCENE_ENGINE_PROTOCOL_VIOLATION 603
#define SCENE_ENGINE_ANIMATION_GROUPS_BROKEN 604

  struct system_methods_t
  {
    int(*init)(system_t* system);
    int(*shutdown)(system_t* system);
    int(*create_scene)(system_t* system, scene_t** scene);
    int(*finalize_scene)(system_t* system, scene_t* scene);
  };

  system_t* system_create(int type);
  void system_free(system_t* system);
  int system_init(system_t* system);
  int system_shutdown(system_t* system);
  int system_create_scene(system_t* system, scene_t** scene);
  int system_finalize_scene(system_t* system, scene_t* scene);

  animation_t* animation_create(int n_groups, int n_frames);
  void animation_free(animation_t* animation);

  task_t* task_create();
  void task_free(task_t* task);

  /// @depricated Use scene engine to create and finalize scene
  scene_t* scene_create();

  void scene_free(scene_t* scene);
  int scene_set_frame(scene_t* scene, int n_frame);
  int scene_relocate(scene_t* scene, int device);
  int scene_static_faces_count(const scene_t* scene);
  int scene_animated_faces_count(const scene_t* scene);

  void host_scene_free_data(scene_t* scene);
  int host_scene_finalize(scene_t* scene);
  int host_scene_set_frame(scene_t* scene, int n_frame);
  int host_scene_relocate(scene_t* scene, int device);

  /**
    @detail 
      1. Make sure meshes does not overlap animation groups.
      2. Not animated geometry goes first.
      3. Groups space cover all faces.
  */
  int host_scene_sort_groups(scene_t* scene);

  float mesh_area(const scene_t* scene, const mesh_t& mesh);
  float mesh_area(const scene_t* scene, int mesh_idx);
  float mesh_volume(const scene_t* scene, const mesh_t& mesh);
  float mesh_volume(const scene_t* scene, int mesh_idx);
  float mesh_C(const scene_t* scene, int mesh_idx);

  /// @brief Calculate areas for all meshes.
  /// @return Total area.
  /// @detail Reallocate target areas array memory.
  float build_meshes_areas(const scene_t* scene, float** areas); 

  /// @brief Calculate areas for all faces.
  /// @detail Reallocate target areas array memory.
  void build_faces_areas(const scene_t* scene, float** areas);

  /// @brief Build face to mesh index.
  /// @detail Reallocate target index array memory.
  void build_face_to_mesh_index(int n_faces, int n_meshes, const mesh_t* meshes, int** index);
  void build_face_to_mesh_index(const scene_t* scene, int** index);

  /// @brief Build face normalized normals.
  /// @detail Reallocate target index array memory.
  void build_face_normals(const scene_t* scene, math::vec3** normals);

  /// @brief Build mesh idx to scene material index (get rid of one level of indirection).
  /// @detail Reallocate target index array memory.
  void build_mesh_material_index(const scene_t* scene, int** index);

  void build_face_animation_group_index(const scene_t* scene, int** index);

  void build_scene_bounding_sphere(scene_t* scene);

  const material_t& mesh_material(const scene_t* scene, int mesh_idx);


  face_graph_index_t* face_graph_index_create(const face_t* faces, int n_faces);
  void face_graph_index_free(face_graph_index_t* index);
  int face_graph_walk_index(const face_graph_index_t* index, face_graph_walker walker, void* param);

  /**
    @brief Walk face graph.
    @note Use face points index to avoid linear scan.
  */
  int face_walk_graph_indexed(const face_t* faces, int n_faces, face_graph_walker walker, void* param);

  /**
    @brief Try to unify face graph normals direction.
    @note It may be impossible to to unify normals for not closed mesh (consider Mobius strip).
    @return Count of flipped faces.
  */
  int face_unify_normals(face_t* faces, int n_faces);

  /**
    @brief Try to unify mesh normals direction.
    @note It may be impossible to to unify normals for not closed mesh (consider Mobius strip).
    @return Count of flipped faces.
  */
  int mesh_unify_normals(scene_t* scene, int mesh_idx);
}
