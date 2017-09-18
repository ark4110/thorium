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

#include "math/operations.h"
#include "math/triangle.h"
#include "emission/system.h"
#include "form_factors/system.h"
#include "emission/malley_cpu.h"
#include "thermal_solution/system.h"
#include "thermal_equation/system.h"
#include "thermal_equation/radiance_cpu.h"
#include "thermal_equation/form_factors.h"
#include "subject/objects.h"

using namespace testing;

const float InitialAccuracy = 0.0025f;
const float CurrentAccuracy = 0.0001f;

template <int SceneEngineID, int EquationEngineID, int EmissionEngineID, int RayCasterEngineID>
struct EnginesConfig
{
  static const int SceneEngine = SceneEngineID;
  static const int EquationEngine = EquationEngineID;
  static const int EmissionEngine = EmissionEngineID;
  static const int RayCasterEngine = RayCasterEngineID;
};

template <typename EnginesConfig>
class RadianceSolution
  : public Test
{
public:
  RadianceSolution()
  {
    SceneEngine = subject::system_create(EnginesConfig::SceneEngine);
    RayCaster = ray_caster::system_create(EnginesConfig::RayCasterEngine);
    
    malley_cpu::params_t emitterParams;
    emitterParams.n_rays = 1000;
    Emitter = emission::system_create(EnginesConfig::EmissionEngine, &emitterParams);

    radiance_equation::params_t equationParams = {};
    equationParams.emitters = &Emitter;
    equationParams.n_emitters = 1;
    equationParams.ray_caster = RayCaster;
    Equation = thermal_equation::system_create(EnginesConfig::EquationEngine, &equationParams);

    thermal_solution::params_t solutionParams = { 1, &Equation };
    System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &solutionParams);

    Materials = (subject::material_t*)calloc(1, sizeof(subject::material_t));
    Materials[0] = subject::black_body();
    Materials[0].front.emission.emissivity = 0;

    subject::face_t* faces = (subject::face_t*)calloc(24, sizeof(subject::face_t));
    memcpy(faces, subject::box(), sizeof(subject::face_t) * 12);
    memcpy(faces + 12, subject::box(), sizeof(subject::face_t) * 12);

    Scale = 3;

    float factor = 1.f / Scale;
    float shift_value = 0.5f - factor / 2.f;
    math::vec3 shift = math::make_vec3(shift_value, shift_value, shift_value);
    for (int f = 0; f != 12; ++f)
    {
      subject::face_t& face = faces[12 + f];
      face.points[0] *= factor;
      face.points[1] *= factor;
      face.points[2] *= factor;

      face.points[0] += shift;
      face.points[1] += shift;
      face.points[2] += shift;

      triangle_flip_normal(face);
    }

    subject::mesh_t* meshes = (subject::mesh_t*)calloc(2, sizeof(subject::mesh_t));
    meshes[0] = { 0, 12, 0 };
    meshes[1] = { 12, 12, 0 };
    system_create_scene(SceneEngine, &Scene);
    Scene->n_faces = 24;
    Scene->faces = faces;
    Scene->n_meshes = 2;
    Scene->meshes = meshes;
    Scene->n_materials = 1;
    Scene->materials = Materials;

    Temperatures[0] = 20.f;
    Temperatures[1] = 300.f;

    Task = thermal_solution::task_create(Scene->n_meshes);
    Task->time_delta = 0.1f;
  }

  ~RadianceSolution()
  {
    thermal_solution::task_free(Task);
    thermal_solution::system_free(System);
    thermal_equation::system_free(Equation);
    emission::system_free(Emitter);
    ray_caster::system_free(RayCaster);
    scene_free(Scene);
    system_free(SceneEngine);
  }
  
  subject::system_t* SceneEngine;
  ray_caster::system_t* RayCaster;
  emission::system_t* Emitter;
  thermal_equation::system_t* Equation;
  thermal_solution::system_t* System;
  subject::material_t* Materials;
  float Scale;
  subject::scene_t* Scene;
  float Temperatures[2];
  thermal_solution::task_t* Task;
};

#if defined _WIN64 || !defined  _WIN32
typedef ::testing::Types<
  EnginesConfig<SCENE_ENGINE_CUDA, THERMAL_EQUATION_RADIANCE_CPU, EMISSION_MALLEY_CPU, RAY_CASTER_OPTIX>,
  EnginesConfig<SCENE_ENGINE_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, EMISSION_MALLEY_CPU, RAY_CASTER_OPTIX>,
  EnginesConfig<SCENE_ENGINE_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, EMISSION_MALLEY_CUDA, RAY_CASTER_OPTIX>
> RadianceEquationTypes;
#else
typedef ::testing::Types<
  EnginesConfig<SCENE_ENGINE_HOST, THERMAL_EQUATION_RADIANCE_CPU, EMISSION_MALLEY_CPU, RAY_CASTER_NAIVE_CPU>
> RadianceEquationTypes;
#endif

TYPED_TEST_CASE(RadianceSolution, RadianceEquationTypes);

TYPED_TEST(RadianceSolution, BoxPreserveEnergy)
{
  using namespace thermal_solution;
  this->Scene->n_faces = 12;
  this->Scene->n_meshes = 1;

  ASSERT_EQ(SCENE_ENGINE_OK, system_finalize_scene(this->SceneEngine, this->Scene));

  const float initialEnergy = this->Temperatures[0];
  int r = 0;
  r = system_set_scene(this->System, this->Scene, this->Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 8; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(this->System, this->Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  ASSERT_NEAR(1.f, this->Task->temperatures[0] / initialEnergy, InitialAccuracy);

  const float currentEnergy = this->Temperatures[0];
  for (int step = 1; step < 42; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  ASSERT_NEAR(1.f, this->Task->temperatures[0] / currentEnergy, CurrentAccuracy);
}

TYPED_TEST(RadianceSolution, NestedBoxesPreserveEnergy)
{
  ASSERT_EQ(SCENE_ENGINE_OK, system_finalize_scene(this->SceneEngine, this->Scene));

  using namespace thermal_solution;
  const float initialEnergy = this->Temperatures[0] * this->Scale * this->Scale + this->Temperatures[1];

  int r = 0;
  r = system_set_scene(this->System, this->Scene, this->Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 8; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(this->System, this->Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);

  float resultEnergy = this->Task->temperatures[0] * this->Scale * this->Scale + this->Task->temperatures[1];
  ASSERT_NEAR(1.f, resultEnergy / initialEnergy, InitialAccuracy);

  const float currentEnergy = this->Temperatures[0] * this->Scale * this->Scale + this->Temperatures[1];
  for (int step = 1; step < 42; ++step)
  {
    r = system_calculate(this->System, this->Task);
  }
  resultEnergy = this->Task->temperatures[0] * this->Scale * this->Scale + this->Task->temperatures[1];
  ASSERT_NEAR(1.f, resultEnergy / currentEnergy, CurrentAccuracy);
}

TYPED_TEST(RadianceSolution, SpecularBoxPreserveEnergy)
{
  using namespace thermal_solution;
  this->Scene->n_faces = 12;
  this->Scene->n_meshes = 1;
  this->Materials[0].rear.optical.absorbance = 0.5;
  this->Materials[0].rear.optical.specular_reflectance = 0.5;

  ASSERT_EQ(SCENE_ENGINE_OK, system_finalize_scene(this->SceneEngine, this->Scene));

  const float initialEnergy = this->Temperatures[0];
  int r = 0;
  r = system_set_scene(this->System, this->Scene, this->Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 8; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(this->System, this->Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  ASSERT_NEAR(1.f, this->Task->temperatures[0] / initialEnergy, InitialAccuracy);

  const float currentEnergy = this->Temperatures[0];
  for (int step = 1; step < 42; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  ASSERT_NEAR(1.f, this->Task->temperatures[0] / currentEnergy, CurrentAccuracy);
}

TYPED_TEST(RadianceSolution, DiffuseBoxPreserveEnergy)
{
  using namespace thermal_solution;
  this->Scene->n_faces = 12;
  this->Scene->n_meshes = 1;
  this->Materials[0].rear.optical.absorbance = 0.5;
  this->Materials[0].rear.optical.diffuse_reflectance = 0.5;

  ASSERT_EQ(SCENE_ENGINE_OK, system_finalize_scene(this->SceneEngine, this->Scene));

  const float initialEnergy = this->Temperatures[0];
  int r = 0;
  r = system_set_scene(this->System, this->Scene, this->Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 8; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(this->System, this->Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  ASSERT_NEAR(1.f, this->Task->temperatures[0] / initialEnergy, InitialAccuracy);

  const float currentEnergy = this->Temperatures[0];
  for (int step = 1; step < 42; ++step)
  {
    r = system_calculate(this->System, this->Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  ASSERT_NEAR(1.f, this->Task->temperatures[0] / currentEnergy, CurrentAccuracy);
}
