// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
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
#include "ballistics/static_prediction.h"

using math::vec3;
using math::make_vec3;
using namespace testing;

template <int SceneEngineID, int EmissionEngineTypeID>
struct BasicConfig
{
  static const int SceneID = SceneEngineID;
  static const int EmissionID = EmissionEngineTypeID;
};

template <int SceneEngineID, int EquationEngineTypeID, int EmissionEngineTypeID, int RayCasterEngineTypeID>
struct EnginesConfig
{
  static const int SceneEngine = SceneEngineID;
  static const int Equation = EquationEngineTypeID;
  static const int Emission = EmissionEngineTypeID;
  static const int RayCaster = RayCasterEngineTypeID;
};

template <typename EngineType>
class MalleyEmission : public Test
{
public:
  MalleyEmission()
  {
    SceneSystem = subject::system_create(EngineType::SceneID);

    malley_cpu::params_t params;
    params.n_rays = 1;
    Emitter = emission::system_create(EngineType::EmissionID, &params);
    
    Scene = ToSubjectScene(MakeFloorScene(SceneSystem));
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });

    system_set_scene(this->Emitter, Scene);
  }

  ~MalleyEmission()
  {
    system_free(Emitter);
    scene_free(Scene);
    system_free(SceneSystem);
  }

  subject::system_t* SceneSystem;
  emission::system_t* Emitter;
  subject::scene_t* Scene;
};

typedef ::testing::Types<BasicConfig<SCENE_ENGINE_HOST, EMISSION_MALLEY_CPU>, BasicConfig<SCENE_ENGINE_CUDA, EMISSION_MALLEY_CUDA>> MalleyEmissionTypes;

TYPED_TEST_CASE(MalleyEmission, MalleyEmissionTypes);

TYPED_TEST(MalleyEmission, AllocateRaycasterTask)
{ 
  auto task = emission::task_create();
  float b[2] = { 1000.f, 1000.f };
  task->mesh_emission = b;
  task->mesh_temperatures = b;
  system_relocate_task(this->Emitter, task, 0);
  system_calculate(this->Emitter, task);
  ASSERT_TRUE(task->rays != 0);
  ASSERT_EQ(4, task->rays->n_tasks);
}

TYPED_TEST(MalleyEmission, ReallocateRaycasterTask)
{
  auto task = emission::task_create();
  float b[2] = { 1000.f, 1000.f };
  task->mesh_emission = b;
  task->mesh_temperatures = b;

  system_relocate_task(this->Emitter, task, 0);
  system_calculate(this->Emitter, task);
  system_calculate(this->Emitter, task);
  ASSERT_TRUE(task->rays != 0);
  ASSERT_EQ(8, task->rays->n_tasks);
}

template <typename EnginesConfig>
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
    auto scene_engine = subject::system_create(EnginesConfig::SceneEngine);
    auto caster = ray_caster::system_create(EnginesConfig::RayCaster);

    auto scene = ToSubjectScene(MakeFloorScene(scene_engine));
    AddMaterial(scene, subject::material_absolute_absorbance());
    AddMesh(scene, { 0, 2, 0 });

    system_finalize_scene(scene_engine, scene);

    parallel_rays_emission_cpu::params_t emitterParams;
    emitterParams.n_sources = SourcesCount;
    emitterParams.sources = Sources;
    auto emitter = emission::system_create(EnginesConfig::Emission, &emitterParams);

    radiance_equation::params_t equationParams = {};
    equationParams.emitters = &emitter;
    equationParams.n_emitters = 1;
    equationParams.ray_caster = caster;
    auto equation = thermal_equation::system_create(EnginesConfig::Equation, &equationParams);

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
    system_free(scene_engine);

    return temperatureDiff;
  }

  int SourcesCount;
  parallel_rays_emission_cpu::parallel_rays_source_t Sources[2];
};

//typedef ::testing::Types<EnginesConfig<EMISSION_PARALLEL_RAYS_CPU, THERMAL_EQUATION_RADIANCE_CPU, RAY_CASTER_NAIVE_CPU>
//  , EnginesConfig<EMISSION_PARALLEL_RAYS_CUDA, THERMAL_EQUATION_RADIANCE_CPU, RAY_CASTER_NAIVE_CPU>
//  , EnginesConfig<EMISSION_PARALLEL_RAYS_CPU, THERMAL_EQUATION_RADIANCE_CUDA, RAY_CASTER_NAIVE_CPU>
//  , EnginesConfig<EMISSION_PARALLEL_RAYS_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, RAY_CASTER_NAIVE_CPU>
//  , EnginesConfig<EMISSION_PARALLEL_RAYS_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, RAY_CASTER_OPTIX>
//  > ParallelEmissionConfigs;

typedef ::testing::Types<
  //EnginesConfig<THERMAL_EQUATION_RADIANCE_CUDA, EMISSION_PARALLEL_RAYS_CUDA, RAY_CASTER_OPTIX>,
  //EnginesConfig<THERMAL_EQUATION_RADIANCE_CPU, EMISSION_PARALLEL_RAYS_CPU, RAY_CASTER_OPTIX>,
  EnginesConfig<SCENE_ENGINE_HOST, THERMAL_EQUATION_RADIANCE_CPU, EMISSION_PARALLEL_RAYS_CPU, RAY_CASTER_NAIVE_CPU>
#if defined _WIN64 || !defined  _WIN32
  , EnginesConfig<SCENE_ENGINE_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, EMISSION_PARALLEL_RAYS_CPU, RAY_CASTER_OPTIX>
#endif
> ParallelEmissionConfigs;

typedef ::testing::Types<BasicConfig<SCENE_ENGINE_HOST, EMISSION_PARALLEL_RAYS_CPU>, BasicConfig<SCENE_ENGINE_CUDA, EMISSION_PARALLEL_RAYS_CUDA>> ParallelEmissionTypes;

TYPED_TEST_CASE(ParallelEmission, ParallelEmissionConfigs);

TYPED_TEST(ParallelEmission, TopSource)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source;
  source.n_rays = 10000;
  source.power = 10;
  source.direction = math::make_vec3(0, 0, -1);

  this->AddSource(source);
  float diff = this->GetTemperatureDiff();
  ASSERT_NEAR(10, diff, 0.2f);
}

TYPED_TEST(ParallelEmission, SkewSource)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source;
  source.n_rays = 10000;
  source.power = 10;
  source.direction = math::make_vec3(0, -1, -1);

  this->AddSource(source);
  float diff = this->GetTemperatureDiff();
  ASSERT_NEAR(5, diff, 0.2f);
}

TYPED_TEST(ParallelEmission, TwoSkewSource)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source;
  source.n_rays = 10000;
  source.power = 10;
  source.direction = math::make_vec3(0, -1, -1);
  this->AddSource(source);

  source.direction = math::make_vec3(-1, 0, 1);
  this->AddSource(source);

  float diff = this->GetTemperatureDiff();
  ASSERT_NEAR(10, diff, 0.2f);
}

template <typename EngineType>
class PlanetEmission : public Test
{
public:
  PlanetEmission(float planet_albedo, float planet_temperature, float planet_radius, float solar_flux, int n_solar_rays)
  {
    SceneSystem = subject::system_create(EngineType::SceneID);

    planet_emission::params_t params = {};

    params.planet_albedo = planet_albedo;
    params.planet_temperature = planet_temperature;
    params.planet_radius = planet_radius;
    params.solar_flux = solar_flux;

    ballistics_static::params_t bparams = {};
    bparams.predictor_func = &static_predictor;
    bparams.func_param = &Prediction;
    params.predictor = ballistics::system_create(BALLISTICS_STATIC, &bparams);

    params.n_subdisivion = 30;
    params.n_rays = 10000;
    params.n_solar_rays = n_solar_rays;
    params.parallel_rays_emitter_type = EngineType::EmissionID;

    Emitter = emission::system_create(EMISSION_PLANET_CPU, &params);

    Scene = ToSubjectScene(MakeFloorScene(SceneSystem));
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });

    system_set_scene(this->Emitter, Scene);
  }

  ~PlanetEmission()
  {
    system_free(Emitter);
    scene_free(Scene);
    system_free(SceneSystem);
  }

  static ballistics::prediction_t static_predictor(void* coords)
  {
    return *static_cast<ballistics::prediction_t*>(coords);
  }

  float kahan_summation(const float* begin, const float* end) {
    float result = 0.f;

    float c = 0.f;
    for (; begin != end; ++begin) {
      float y = *begin - c;
      float t = result + y;
      c = (t - result) - y;
      result = t;
    }
    return result;
  }

  float calculate_flux()
  {
    auto task = emission::task_create();
    float b[2] = { 1000.f, 1000.f };
    task->mesh_emission = b;
    task->mesh_temperatures = b;
    system_calculate(this->Emitter, task);
    system_relocate_task(this->Emitter, task, -1);

    float total_flux = kahan_summation(task->rays->power, task->rays->power + task->rays->n_tasks);

    math::sphere_t sphere = math::triangles_bsphere(Scene->faces, Scene->n_faces);
    float bsphere_scale = parallel_rays_emission_cpu::scene_bsphere_emission_scale;
    float effective_area = float(M_PI) * sphere.radius * sphere.radius * bsphere_scale * bsphere_scale;
    float flux = total_flux / effective_area;

    return flux;
  }

  subject::system_t* SceneSystem;
  emission::system_t* Emitter;
  subject::scene_t* Scene;
  ballistics::prediction_t Prediction;
};

static const float earth_polar_radius = 6.3568e6f;
static const float sun_distance = 1.495978e11f;

template <typename EngineType>
class SolarIrradiance : public PlanetEmission<EngineType>
{
public:
  SolarIrradiance()
    : PlanetEmission<EngineType>(0.34f, 0, earth_polar_radius, 1353, 0)
  {
  }
};

template <typename EngineType>
class EarthSelfEmission : public PlanetEmission<EngineType>
{
public:
  EarthSelfEmission()
    : PlanetEmission<EngineType>(0.34f, 250, earth_polar_radius, 0, 0)
  {
  }
};

template <typename EngineType>
class SolarRadiance : public PlanetEmission<EngineType>
{
public:
  SolarRadiance()
    : PlanetEmission<EngineType>(0.f, 0, earth_polar_radius, 1353, 100)
  {
  }
};

TYPED_TEST_CASE(SolarIrradiance, ParallelEmissionTypes);
TYPED_TEST_CASE(EarthSelfEmission, ParallelEmissionTypes);
TYPED_TEST_CASE(SolarRadiance, ParallelEmissionTypes);

TYPED_TEST(SolarIrradiance, Alt1300Theta0)
{
  const float altitude = 1300e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);  
  this->Prediction.sun_position = math::make_vec3(0, 0, sun_distance - altitude);
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(400, flux, 5.f);
}

TYPED_TEST(SolarIrradiance, Alt1690Theta30)
{
  const float altitude = 1690e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);  
  this->Prediction.sun_position = math::make_vec3(sinf(float(M_PI) / 6.f) * (sun_distance - altitude), 0, cosf(float(M_PI) / 6.f) * (sun_distance - altitude));
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(300, flux, 5.f);
}

TYPED_TEST(SolarIrradiance, Alt200Theta60)
{
  const float altitude = 200e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  this->Prediction.sun_position = math::make_vec3(sinf(float(M_PI) / 3.f) * (sun_distance - altitude), 0, cosf(float(M_PI) / 3.f) * (sun_distance - altitude));
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(350, flux, 5.f);
}

TYPED_TEST(SolarIrradiance, Alt3200Theta60)
{
  const float altitude = 3200e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  this->Prediction.sun_position = math::make_vec3(sinf(float(M_PI) / 3.f) * sun_distance, 0, cosf(float(M_PI) / 3.f) * sun_distance - earth_polar_radius - altitude);
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(110, flux, 5.f);
}

TYPED_TEST(SolarIrradiance, Alt400Theta80)
{
  const float altitude = 400e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  this->Prediction.sun_position = math::make_vec3(sinf(float(M_PI) / 180 * 80) * sun_distance, 0, cosf(float(M_PI) / 180 * 80) * sun_distance - earth_polar_radius - altitude);
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(101, flux, 5.f);
}

TYPED_TEST(SolarIrradiance, Alt2000Theta90)
{
  const float altitude = 2000e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  this->Prediction.sun_position = math::make_vec3(sun_distance, 0, 0);
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(21, flux, 5.f);
}

TYPED_TEST(EarthSelfEmission, Alt600)
{
  const float altitude = 600e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  this->Prediction.sun_position = math::make_vec3(sun_distance, 0, 0);
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(260, flux, 5.f);
}

TYPED_TEST(SolarRadiance, IsConsidered)
{
  const float altitude = 600e3;

  this->Prediction.earth_position = math::make_vec3(0, 0, -earth_polar_radius - altitude);
  this->Prediction.sun_position = math::make_vec3(sun_distance, 0, 0);
  this->Prediction.eclispsed = 0;

  float flux = this->calculate_flux();
  ASSERT_NEAR(1353, flux, 1.f);
}
