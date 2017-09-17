#include "test_scene.h"

#include <memory>

namespace testing
{
  subject::scene_t* MakeEmptyScene(subject::system_t* engine)
  {
    subject::scene_t* scene;
    system_create_scene(engine, &scene);
    system_finalize_scene(engine, scene);
    return scene;
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

  subject::scene_t* MakeSceneFromFaces(subject::system_t* engine, int n_faces, ray_caster::face_t* faces)
  {
    subject::scene_t* scene = 0;
    system_create_scene(engine, &scene);
    scene->n_faces = n_faces;
    scene->faces = faces;
    system_finalize_scene(engine, scene);
    return scene;
  }

  subject::scene_t* ToSubjectScene(subject::scene_t* rcs)
  {
    return rcs;
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

  subject::scene_t* MakeFloorScene(subject::system_t* engine)
  {
    int n_faces = 2;
    ray_caster::face_t* faces = (ray_caster::face_t*)malloc(n_faces * sizeof(ray_caster::face_t));
    faces[0] = make_floor_face1();
    faces[1] = make_floor_face2();

    return MakeSceneFromFaces(engine, n_faces, faces);
  }

  subject::scene_t* MakeStackScene(subject::system_t* engine)
  {
    int n_faces = 2;
    ray_caster::face_t* faces = (ray_caster::face_t*)malloc(n_faces * sizeof(ray_caster::face_t));
    faces[0] = make_floor_face1();
    faces[1] = make_stack_face();

    return MakeSceneFromFaces(engine, n_faces, faces);
  }
}