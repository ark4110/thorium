#pragma once

#include <ray_caster/system.h>

namespace testing
{
  ray_caster::scene_t* MakeEmptyScene();
  ray_caster::face_t make_floor_face1();
  ray_caster::face_t make_floor_face2();
  ray_caster::face_t make_stack_face();

  ray_caster::scene_t* MakeSceneFromFaces(int n_faces, ray_caster::face_t* faces);
  ray_caster::scene_t* MakeFloorScene();
  ray_caster::scene_t* MakeStackScene();
}
