#include "test_scene.h"

#include <memory>

namespace testing
{
  ray_caster::scene_t* MakeEmptyScene()
  {
    return ray_caster::scene_create();
  }

  ray_caster::face_t make_floor_face1()
  {
    math::vec3 a = { 0.f, 0.f, 0.f };
    math::vec3 b = { 1.f, 0.f, 0.f };
    math::vec3 c = { 0.f, 1.f, 0.f };

    return make_face(a, b, c);
  }

  ray_caster::face_t make_floor_face2()
  {
    math::vec3 a = { 1.f, 0.f, 0.f };
    math::vec3 b = { 0.f, 1.f, 0.f };
    math::vec3 c = { 1.f, 1.f, 0.f };

    return make_face(a, b, c);
  }

  ray_caster::face_t make_stack_face()
  {
    math::vec3 a = { 0.f, 0.f, -1.f };
    math::vec3 b = { 1.f, 0.f, -1.f };
    math::vec3 c = { 0.f, 1.f, -1.f };

    return make_face(a, b, c);
  }

  ray_caster::scene_t* MakeSceneFromFaces(int n_faces, ray_caster::face_t* faces)
  {
    ray_caster::scene_t* s = ray_caster::scene_create();
    *s = { /*.n_faces =*/ n_faces, /*.faces =*/ faces };
    return s;
  }

  ray_caster::scene_t* MakeFloorScene()
  {
    int n_faces = 2;
    ray_caster::face_t* faces = (ray_caster::face_t*)malloc(n_faces * sizeof(ray_caster::face_t));
    faces[0] = make_floor_face1();
    faces[1] = make_floor_face2();

    return MakeSceneFromFaces(n_faces, faces);
  }

  ray_caster::scene_t* MakeStackScene()
  {
    int n_faces = 2;
    ray_caster::face_t* faces = (ray_caster::face_t*)malloc(n_faces * sizeof(ray_caster::face_t));
    faces[0] = make_floor_face1();
    faces[1] = make_stack_face();

    return MakeSceneFromFaces(n_faces, faces);
  }
}