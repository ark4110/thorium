#include "test_scene.h"

#include <memory>

namespace testing
{
  ray_caster::scene_t* MakeEmptyScene()
  {
    return subject::scene_create();
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
    ray_caster::scene_t* s = subject::scene_create();
    *s = { /*.n_faces =*/ n_faces, /*.faces =*/ faces };
    return s;
  }

  subject::scene_t* ToSubjectScene(ray_caster::scene_t* rcs)
  {
    subject::scene_t* ss = subject::scene_create();
    ss->faces = rcs->faces;
    ss->n_faces = rcs->n_faces;

    rcs->faces = 0;
    scene_free(rcs);

    return ss;
  }

  void AddMaterial(subject::scene_t* scene, subject::material_t material)
  {
    scene->materials = (subject::material_t*)realloc(scene->materials, (1 + scene->n_materials) * sizeof(subject::material_t));
    scene->materials[scene->n_materials] = material;
    ++scene->n_materials;
  }

  void AddMesh(subject::scene_t* scene, subject::mesh_t mesh)
  {
    scene->meshes = (subject::mesh_t*)realloc(scene->meshes, (1 + scene->n_meshes) * sizeof(subject::mesh_t));
    scene->meshes[scene->n_meshes] = mesh;
    ++scene->n_meshes;
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