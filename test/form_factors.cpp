// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#include "gtest/gtest.h"
#include "test_scene.h"
#include <ray_caster/system.h>
#include <form_factors/system.h>
#include <emission/system.h>
#include <emission/malley_cpu.h>
#include <math/operations.h>
#include <math/triangle.h>
#include <math/mat.h>

using math::vec3;
using math::make_vec3;
using namespace testing;
using namespace form_factors;
using subject::scene_t;
using subject::mesh_t;

template <int SceneEngineID, int EmitterTypeID, int RayCasterTypeID>
struct EngineType
{
  static const int SceneEngineID = SceneEngineID;
  static const int EmitterID = EmitterTypeID;
  static const int RayCasterID = RayCasterTypeID;
};

#if defined _WIN64 || !defined  _WIN32
#define RAY_CASTER_SELECTOR RAY_CASTER_OPTIX
#else
#define RAY_CASTER_SELECTOR RAY_CASTER_NAIVE_CUDA
#endif

typedef ::testing::Types<
  EngineType<SCENE_ENGINE_CUDA, EMISSION_MALLEY_CPU, RAY_CASTER_SELECTOR>,
  EngineType<SCENE_ENGINE_CUDA, EMISSION_MALLEY_CUDA, RAY_CASTER_SELECTOR>
> FormFactorsTypes;

template <typename EngineType>
class FormFactors : public Test
{
public:
  FormFactors()
  {
    SceneSystem = subject::system_create(EngineType::SceneEngineID);
    Scene = 0;
    malley_cpu::params_t emitter_params;
    emitter_params.n_rays = 40000;
    Emitter = emission::system_create(EngineType::EmitterID, &emitter_params);
    RayCaster = ray_caster::system_create(EngineType::RayCasterID);
    form_factors::params_t params{ Emitter, RayCaster };
    Calculator = system_create(FORM_FACTORS_CPU, &params);
  }

  ~FormFactors()
  {
    system_free(Calculator);
    system_free(RayCaster);
    system_free(Emitter);
    scene_free(Scene);
    system_free(SceneSystem);
  }

  face_t make_floor_face1(math::vec3 offset)
  {
    vec3 a = { 0.f, 0.f, 0.f };
    vec3 b = { 1.f, 0.f, 0.f };
    vec3 c = { 0.f, 1.f, 0.f };

    return make_face(a + offset, b + offset, c + offset);
  }

  face_t make_floor_face2(math::vec3 offset)
  {
    vec3 a = { 1.f, 0.f, 0.f };
    vec3 b = { 1.f, 1.f, 0.f };
    vec3 c = { 0.f, 1.f, 0.f };

    return make_face(a + offset, b + offset, c + offset);
  }

  face_t make_wall_face1(math::vec3 offset)
  {
    vec3 a = { 0.f, 0.f, 0.f };
    vec3 b = { 0.f, 1.f, 0.f };
    vec3 c = { 0.f, 0.f, 1.f };

    return make_face(a + offset, b + offset, c + offset);
  }

  face_t make_wall_face2(math::vec3 offset)
  {
    vec3 a = { 0.f, 1.f, 0.f };
    vec3 b = { 0.f, 0.f, 1.f };
    vec3 c = { 0.f, 1.f, 1.f };

    return make_face(a + offset, b + offset, c + offset);
  }

  scene_t* MakeParallelPlanesScene(float c, math::mat33 rotation)
  {
    int n_faces = 4;
    face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
    faces[0] = make_floor_face1(math::make_vec3(0, 0, 0));
    faces[1] = make_floor_face2(math::make_vec3(0, 0, 0));
    faces[2] = make_floor_face1(math::make_vec3(0, 0, c));
    faces[3] = make_floor_face2(math::make_vec3(0, 0, c));

    for (int i = 0; i != n_faces; ++i)
    {
      faces[i].points[0] = rotation * faces[i].points[0];
      faces[i].points[1] = rotation * faces[i].points[1];
      faces[i].points[2] = rotation * faces[i].points[2];
    }
    
    Scene = MakeSceneFromFaces(SceneSystem, n_faces, faces);
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });
    AddMesh(Scene, { 2, 2, 0 });

    system_finalize_scene(SceneSystem, Scene);

    return Scene;
  }

  scene_t* MakePerpendicularPlanesScene(math::mat33 rotation)
  {
    int n_faces = 4;
    face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
    faces[0] = make_floor_face1(math::make_vec3(0, 0, 0));
    faces[1] = make_floor_face2(math::make_vec3(0, 0, 0));
    faces[2] = make_wall_face1(math::make_vec3(0, 0, 0));
    faces[3] = make_wall_face2(math::make_vec3(0, 0, 0));

    for (int i = 0; i != n_faces; ++i)
    {
      faces[i].points[0] = rotation * faces[i].points[0];
      faces[i].points[1] = rotation * faces[i].points[1];
      faces[i].points[2] = rotation * faces[i].points[2];
    }

    Scene = ToSubjectScene(MakeSceneFromFaces(SceneSystem, n_faces, faces));
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });
    AddMesh(Scene, { 2, 2, 0 });

    system_finalize_scene(SceneSystem, Scene);

    return Scene;
  }

  scene_t* MakeBoxScene(math::mat33 rotation)
  {
    math::vec3 v[] = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } };
    int n_faces = 12;
    face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
    faces[0] = make_face(v[0], v[1], v[2]);
    faces[1] = make_face(v[0], v[3], v[2]);    
    faces[8] = make_face(v[0], v[1], v[5]);
    faces[9] = make_face(v[0], v[4], v[5]);
    faces[4] = make_face(v[0], v[3], v[7]);
    faces[5] = make_face(v[0], v[4], v[7]);
    faces[6] = make_face(v[1], v[2], v[6]);
    faces[7] = make_face(v[1], v[5], v[6]);
    faces[2] = make_face(v[3], v[2], v[6]);
    faces[3] = make_face(v[3], v[7], v[6]);
    faces[10] = make_face(v[4], v[5], v[6]);
    faces[11] = make_face(v[4], v[7], v[6]);
    

    for (int i = 0; i != n_faces; ++i)
    {
      faces[i].points[0] = rotation * faces[i].points[0];
      faces[i].points[1] = rotation * faces[i].points[1];
      faces[i].points[2] = rotation * faces[i].points[2];
    }

    Scene = MakeSceneFromFaces(SceneSystem, n_faces, faces);
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, n_faces, 0 });

    system_finalize_scene(SceneSystem, Scene);

    return Scene;
  }

  subject::system_t* SceneSystem;
  scene_t* Scene;
  ray_caster::system_t* RayCaster;
  emission::system_t* Emitter;
  system_t* Calculator;
};

TYPED_TEST_CASE(FormFactors, FormFactorsTypes);

math::point_t theor_parallel_planes(math::point_t a, math::point_t b, math::point_t c) {
  auto x = a / c, y = b / c, xq = x * x, yq = y * y;
  auto result = 2 / math::point_t(M_PI) / x / y * (logf(sqrtf((1 + xq) * (1 + yq) / (1 + xq + yq))) + x * sqrtf(1 + yq) * atanf(x / sqrtf(1 + yq)) + y * sqrtf(1 + xq) * atanf(y / sqrtf(1 + xq)) - x * atanf(x) - y * atanf(y));
  return result / 2;
}

math::point_t theor_perpendicular_planes(double H, double W, double L) {
  double h = H / L;
  double w = W / L;
  auto a = (1 + h * h) * (1 + w * w) / (1 + h * h + w * w);
  auto b = w * w * (1 + h * h + w * w) / (1 + w * w) / (h * h + w * w);
  auto c = h * h * (1 + h * h + w * w) / (1 + h * h) / (h * h + w * w);
  auto result = 0.5 / M_PI / w * (h * atan(1 / h) + w * atan(1 / w) - sqrt(h * h + w * w) * atan(1 / sqrt(h * h + w * w)) + 0.25 * log(a * pow(b, w * w) * pow(c, h * h)));
  return math::point_t(result);
}

TYPED_TEST(FormFactors, ParallelPlanesCorrect)
{
  float distances[] = { .5f, 1, 2, 3, 10 };
  for (int i = 0; i != sizeof(distances) / sizeof(float); ++i)
  {
    scene_t* stackScene = this->MakeParallelPlanesScene(distances[i], math::IDENTITY_33);

    system_set_scene(this->Calculator, stackScene);

    float factors[2 * 2];
    task_t task = { factors };

    float theoretical = theor_parallel_planes(1, 1, distances[i]);

    ASSERT_EQ(FORM_FACTORS_OK, system_calculate(this->Calculator, &task));
    EXPECT_NEAR(theoretical, factors[1], 0.01);
    EXPECT_NEAR(theoretical, factors[2], 0.01);
  }
}

TYPED_TEST(FormFactors, RotatedParallelPlanesCorrect)
{
  float distances[] = { .5f, 1, 2, 3, 10 };
  for (int i = 0; i != sizeof(distances) / sizeof(float); ++i)
  {
    math::mat33 rotation = math::axis_rotation(float(M_PI) / 4, float(M_PI) / 4, float(M_PI) / 4);
    scene_t* stackScene = this->MakeParallelPlanesScene(distances[i], rotation);

    system_set_scene(this->Calculator, stackScene);

    float factors[2 * 2];
    task_t task = { factors };

    float theoretical = theor_parallel_planes(1, 1, distances[i]);

    ASSERT_EQ(FORM_FACTORS_OK, system_calculate(this->Calculator, &task));
    EXPECT_NEAR(theoretical, factors[1], 0.01);
    EXPECT_NEAR(theoretical, factors[2], 0.01);
  }
}

TYPED_TEST(FormFactors, PerpendicularPlanesCorrect)
{
  scene_t* stackScene = this->MakePerpendicularPlanesScene(math::IDENTITY_33);

  system_set_scene(this->Calculator, stackScene);

  float factors[2 * 2];
  task_t task = { factors };

  float theoretical = theor_perpendicular_planes(1, 1, 1);

  ASSERT_EQ(FORM_FACTORS_OK, system_calculate(this->Calculator, &task));
  EXPECT_NEAR(theoretical, factors[2], 0.01);
}

TYPED_TEST(FormFactors, RotatedPerpendicularPlanesCorrect)
{
  math::mat33 rotation = math::axis_rotation(float(M_PI) / 4, float(M_PI) / 4, float(M_PI) / 4);
  scene_t* stackScene = this->MakePerpendicularPlanesScene(rotation);

  system_set_scene(this->Calculator, stackScene);
  
  float factors[2 * 2];
  task_t task = { factors };

  float theoretical = theor_perpendicular_planes(1, 1, 1);

  ASSERT_EQ(FORM_FACTORS_OK, system_calculate(this->Calculator, &task));
  EXPECT_NEAR(theoretical, factors[2], 0.01);
}

TYPED_TEST(FormFactors, BoxCorrect)
{
  scene_t* scene = this->MakeBoxScene(math::IDENTITY_33);

  system_set_scene(this->Calculator, scene);

  float factors[1 * 1];
  task_t task = { factors };

  float theoretical = 0.5f;

  ASSERT_EQ(FORM_FACTORS_OK, system_calculate(this->Calculator, &task));
  EXPECT_NEAR(theoretical, factors[0], 0.01);
}