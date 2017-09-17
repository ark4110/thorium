#pragma once

#include <ray_caster/system.h>
#include <subject/system.h>

namespace testing
{
  subject::scene_t* MakeEmptyScene();
  ray_caster::face_t make_floor_face1();
  ray_caster::face_t make_floor_face2();
  ray_caster::face_t make_stack_face();

  subject::scene_t* MakeSceneFromFaces(int n_faces, ray_caster::face_t* faces);
  subject::scene_t* MakeFloorScene();
  subject::scene_t* MakeStackScene();

  subject::scene_t* ToSubjectScene(subject::scene_t*);
  void AddMaterial(subject::scene_t* scene, subject::material_t material);
  void AddMesh(subject::scene_t* scene, subject::mesh_t mesh);
}
