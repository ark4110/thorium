#pragma once

#include <ray_caster/system.h>
#include <subject/system.h>

namespace testing
{
  ray_caster::scene_t* MakeEmptyScene();
  ray_caster::face_t make_floor_face1();
  ray_caster::face_t make_floor_face2();
  ray_caster::face_t make_stack_face();

  ray_caster::scene_t* MakeSceneFromFaces(int n_faces, ray_caster::face_t* faces);
  ray_caster::scene_t* MakeFloorScene();
  ray_caster::scene_t* MakeStackScene();

  subject::scene_t* ToSubjectScene(ray_caster::scene_t*);
  void AddMaterial(subject::scene_t* scene, subject::material_t material);
  void AddMesh(subject::scene_t* scene, subject::mesh_t mesh);
}
