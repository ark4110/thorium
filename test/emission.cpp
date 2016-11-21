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
#include "emission/malley_cpu.h"
#include "emission/parallel_rays_cpu.h"
#include "emission/planet_emission.h"
#include "thermal_solution/system.h"
#include "thermal_equation/system.h"
#include "thermal_equation/radiance_cpu.h"
#include "math/operations.h"
#include "math/triangle.h"
#include "subject/objects.h"

using math::vec3;
using math::make_vec3;
using namespace testing;

template <int EngineTypeID>
struct EngineType
{
  static const int ID = EngineTypeID;
};

template <typename EngineType>
class MalleyEmission : public Test
{
public:
  MalleyEmission()
  {
    malley_cpu::params_t params;
    params.n_rays = 1;
    Emitter = emission::system_create(EngineType::ID, &params);
    
    Scene = ToSubjectScene(MakeFloorScene());
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });

    system_set_scene(Emitter, Scene);
  }

  ~MalleyEmission()
  {
    system_free(Emitter);
    scene_free(Scene);
  }

  emission::system_t* Emitter;
  subject::scene_t* Scene;
};

typedef ::testing::Types<EngineType<EMISSION_MALLEY_CPU>, EngineType<EMISSION_MALLEY_CUDA>> MalleyEmissionTypes;

TYPED_TEST_CASE(MalleyEmission, MalleyEmissionTypes);

TYPED_TEST(MalleyEmission, AllocateRaycasterTask)
{ 
  auto task = emission::task_create();
  float b[2] = { 1000.f, 1000.f };
  task->mesh_emission = b;
  task->mesh_temperatures = b;
  system_calculate(Emitter, task);
  ASSERT_TRUE(task->rays != 0);
  ASSERT_EQ(4, task->rays->n_tasks);
}

TYPED_TEST(MalleyEmission, ReallocateRaycasterTask)
{
  auto task = emission::task_create();
  float b[2] = { 1000.f, 1000.f };
  task->mesh_emission = b;
  task->mesh_temperatures = b;

  system_calculate(Emitter, task);
  system_calculate(Emitter, task);
  ASSERT_TRUE(task->rays != 0);
  ASSERT_EQ(8, task->rays->n_tasks);
}

TYPED_TEST(MalleyEmission, DISABLED_LargeObject)
{
  const int size = 119;
  subject::scene_t* scene = subject::scene_create();
  scene->faces = subject::plane_grid_faces(100, 100, size, size);
  scene->n_faces = size * size * 2;

  scene->meshes = subject::plane_grid_meshes(size, size);
  scene->n_meshes = size * size;

  AddMaterial(scene, subject::black_body());
  
  auto task = emission::task_create();
  float* temperatures = (float*)malloc(sizeof(float) * scene->n_meshes);
  task->mesh_temperatures = temperatures;
  task->mesh_emission = (float*)malloc(sizeof(float) * scene->n_meshes);

  for (int i = 0; i != size * size; ++i)
  {
    scene->meshes[i].material_idx = 0;
    temperatures[i] = 300.f * i /size / size;
  }
  
  system_set_scene(Emitter, scene);
  for (int i = 0; i != 3; ++i)
  {
    system_calculate(Emitter, task);
    task_reset(task->rays);
  }

  free(task->mesh_emission);
  free(temperatures);

  scene_free(scene);
}


template <typename EngineType>
class ParallelEmission : public Test
{
public:
  ParallelEmission()
    : SourcesCount(0)
  {
  }

  void AddSource(parallel_rays_emission_cpu::parallel_rays_source_t source)
  {
    Sources[SourcesCount++] = source;
  }

  float GetTemperatureDiff()
  {
    auto caster = ray_caster::system_create(RAY_CASTER_NAIVE_CPU);

    auto scene = ToSubjectScene(MakeFloorScene());
    AddMaterial(scene, subject::material_absolute_absorbance());
    AddMesh(scene, { 0, 2, 0 });

    parallel_rays_emission_cpu::params_t emitterParams;
    emitterParams.n_sources = SourcesCount;
    emitterParams.sources = Sources;
    auto emitter = emission::system_create(EngineType::ID, &emitterParams);

    radiance_equation::params_t equationParams = {};
    equationParams.emitters = &emitter;
    equationParams.n_emitters = 1;
    equationParams.ray_caster = caster;
    auto equation = thermal_equation::system_create(THERMAL_EQUATION_RADIANCE_CPU, &equationParams);

    thermal_solution::params_t solutionParams = { 1, &equation };
    auto solution = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &solutionParams);
    
    float* temperatures = (float*)malloc(sizeof(float));
    temperatures[0] = 300;

    system_set_scene(solution, scene, temperatures);

    auto task = thermal_solution::task_create(scene->n_meshes);
    task->time_delta = 1;
    task->n_step = 1;
    task->temperatures = temperatures;
    task->time_delta = 1.f;

    thermal_solution::system_calculate(solution, task);

    float temperatureDiff = task->temperatures[0] - 300.f;
    task_free(task);
    system_free(solution);
    system_free(equation);
    system_free(emitter);
    system_free(caster);
    scene_free(scene);

    return temperatureDiff;
  }

  int SourcesCount;
  parallel_rays_emission_cpu::parallel_rays_source_t Sources[2];
};

typedef ::testing::Types<EngineType<EMISSION_PARALLEL_RAYS_CPU>, EngineType<EMISSION_PARALLEL_RAYS_CUDA>> ParallelEmissionTypes;

TYPED_TEST_CASE(ParallelEmission, ParallelEmissionTypes);

TYPED_TEST(ParallelEmission, TopSource)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source;
  source.n_rays = 10000;
  source.power = 10;
  source.direction = math::make_vec3(0, 0, -1);

  AddSource(source);
  float diff = GetTemperatureDiff();
  ASSERT_NEAR(10, diff, 0.2f);
}

TYPED_TEST(ParallelEmission, SkewSource)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source;
  source.n_rays = 10000;
  source.power = 10;
  source.direction = math::make_vec3(0, -1, -1);

  AddSource(source);
  float diff = GetTemperatureDiff();
  ASSERT_NEAR(5, diff, 0.2f);
}

TYPED_TEST(ParallelEmission, TwoSkewSource)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source;
  source.n_rays = 10000;
  source.power = 10;
  source.direction = math::make_vec3(0, -1, -1);
  AddSource(source);

  source.direction = math::make_vec3(-1, 0, 1);
  AddSource(source);

  float diff = GetTemperatureDiff();
  ASSERT_NEAR(10, diff, 0.2f);
}

class PlanetEmission : public Test
{
public:
  PlanetEmission()
  {
    planet_emission::params_t params;

    params.planet_albedo = 0.34f;
    params.planet_radius = 6.3568e6f; // polar radius
    params.solar_flux = 1353;

    params.predictor = { &static_predictor, &Prediction };

    params.n_subdisivion = 5;
    params.n_rays = 10000;
    params.parallel_rays_emitter_type = EMISSION_PARALLEL_RAYS_CPU;

    Emitter = emission::system_create(EMISSION_PLANET_CPU, &params);

    Scene = ToSubjectScene(MakeFloorScene());
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });

    system_set_scene(Emitter, Scene);
  }

  ~PlanetEmission()
  {
    system_free(Emitter);
    scene_free(Scene);
  }

  static ballistics::prediction_t static_predictor(void* coords)
  {
    return *static_cast<ballistics::prediction_t*>(coords);
  }

  float calculate_flux()
  {
    auto task = emission::task_create();
    float b[2] = { 1000.f, 1000.f };
    task->mesh_emission = b;
    task->mesh_temperatures = b;
    system_calculate(Emitter, task);
    float total_flux = 0;
    for (int i = 0; i != task->rays->n_tasks; ++i)
      total_flux += task->power[i];

    math::sphere_t sphere = math::triangles_bsphere(Scene->faces, Scene->n_faces);
    float bsphere_scale = parallel_rays_emission_cpu::scene_bsphere_emission_scale;
    float effective_area = M_PI * sphere.radius * sphere.radius * bsphere_scale * bsphere_scale;
    float flux = total_flux / effective_area;

    return flux;
  }

  emission::system_t* Emitter;
  subject::scene_t* Scene;
  ballistics::prediction_t Prediction;
};

static const float earth_polar_radius = 6.3568e6f;

TEST_F(PlanetEmission, Alt1300Theta0)
{
  const float altitude = 1300e3;
  const float sun_distance = 1.495978e11f;

  Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);  
  Prediction.sun_position = math::make_vec3(0, 0, sun_distance - altitude);
  Prediction.eclispsed = 0;

  float flux = calculate_flux();
  ASSERT_NEAR(400, flux, 10.f);
}

TEST_F(PlanetEmission, Alt1690Theta30)
{
  const float altitude = 1690e3;
  const float sun_distance = 1.495978e11f;

  Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);  
  Prediction.sun_position = math::make_vec3(sinf(M_PI / 6.f) * (sun_distance - altitude), 0, cosf(M_PI / 6.f) * (sun_distance - altitude));
  Prediction.eclispsed = 0;

  float flux = calculate_flux();
  ASSERT_NEAR(300, flux, 10.f);
}

TEST_F(PlanetEmission, Alt200Theta60)
{
  const float altitude = 200e3;
  const float sun_distance = 1.495978e11f;

  Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  Prediction.sun_position = math::make_vec3(sinf(M_PI / 3.f) * (sun_distance - altitude), 0, cosf(M_PI / 3.f) * (sun_distance - altitude));
  Prediction.eclispsed = 0;

  float flux = calculate_flux();
  ASSERT_NEAR(350, flux, 10.f);
}

TEST_F(PlanetEmission, Alt3200Theta60)
{
  const float altitude = 3200e3;
  const float sun_distance = 1.495978e11f;

  Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  Prediction.sun_position = math::make_vec3(sinf(M_PI / 3.f) * sun_distance, 0, cosf(M_PI / 3.f) * sun_distance - earth_polar_radius - altitude);
  Prediction.eclispsed = 0;

  float flux = calculate_flux();
  ASSERT_NEAR(110, flux, 10.f);
}

TEST_F(PlanetEmission, Alt400Theta80)
{
  const float altitude = 400e3;
  const float sun_distance = 1.495978e11f;

  Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  Prediction.sun_position = math::make_vec3(sinf(M_PI / 180 * 80) * sun_distance, 0, cosf(M_PI / 180 * 80) * sun_distance - earth_polar_radius - altitude);
  Prediction.eclispsed = 0;

  float flux = calculate_flux();
  ASSERT_NEAR(100, flux, 10.f);
}
