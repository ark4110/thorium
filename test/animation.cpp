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
#include "../import_export/obj_import.h"
#include "../math/mat.h"
#include "../subject/system.h"
#include "../subject/operations.h"
#include "../emission/system.h"
#include "../emission/malley_cpu.h"
#include "../emission/parallel_rays_cpu.h"
#include "../ray_caster/system.h"
#include "../thermal_equation/system.h"
#include "../thermal_equation/radiance_cpu.h"

using namespace testing;

float diff(float a, float b)
{
  return fabsf(a - b) / a / b;
}

/**
  @detail Test hierarchy scene consists of three object: standalone cube at the left, parent cube at the center and child cube at the right.
*/

TEST(AnimationImport, BasicImport)
{
  subject::system_t* scene_system = subject::system_create(SCENE_ENGINE_HOST);
  subject::scene_t* scene = 0;
  int r = 0;
  r = subject::system_create_scene(scene_system, &scene);
  ASSERT_TRUE(r == SCENE_ENGINE_OK);
  r = obj_import::scene("models/test_hierarchy.obj", scene);
  ASSERT_TRUE(r == OBJ_IMPORT_OK);
  
  ASSERT_EQ(3, scene->n_meshes);
  ASSERT_EQ(36, scene->n_faces);
  ASSERT_STREQ("Standalone", scene->meshes[0].name);

  // Child lexicographically less Parent, so goes first.
  ASSERT_STREQ("Child", scene->meshes[1].name);
  ASSERT_STREQ("Parent", scene->meshes[2].name);

  ASSERT_EQ(1, scene->animation->n_groups);
  ASSERT_EQ(12, scene->animation->groups->n_offset);
  ASSERT_EQ(24, scene->animation->groups->n_length);

  scene_free(scene);
  system_free(scene_system);
}

TEST(CudaSceneEngine, Relocation)
{
  auto system = subject::system_create(SCENE_ENGINE_CUDA);
  subject::scene_t* scene = 0;
  system_create_scene(system, &scene);
  ASSERT_EQ(OBJ_IMPORT_OK, obj_import::scene("models/test_hierarchy.obj", scene));
  system_finalize_scene(system, scene);
  scene_relocate(scene, 0);
  scene_relocate(scene, -1);

  scene_free(scene);
  system_free(system);
}

template <int EngineTypeID>
struct SceneEngineType
{
  static const int ID = EngineTypeID;
};

template <typename SceneEngineType>
class ImportedAnimatedScene : public Test
{
public:
  ImportedAnimatedScene()
  {
    SceneSystem = subject::system_create(SceneEngineType::ID);
    system_create_scene(SceneSystem, &Scene);
    obj_import::scene("models/test_hierarchy.obj", Scene);
  }

  ~ImportedAnimatedScene()
  {
    scene_free(Scene);
    system_free(SceneSystem);
  }

public:
  subject::system_t* SceneSystem;
  subject::scene_t* Scene;
};

typedef ::testing::Types<SceneEngineType<SCENE_ENGINE_HOST>, SceneEngineType<SCENE_ENGINE_CUDA>> SceneEnginesTypes;

TYPED_TEST_CASE(ImportedAnimatedScene, SceneEnginesTypes);

TYPED_TEST(ImportedAnimatedScene, FinalizationTest)
{
  int r = 0;
  r = system_finalize_scene(this->SceneSystem, this->Scene);
  ASSERT_EQ(SCENE_ENGINE_OK, r);
  ASSERT_NE(nullptr, this->Scene->face_areas);
  ASSERT_NE(nullptr, this->Scene->face_normals);
  ASSERT_NE(nullptr, this->Scene->face_to_mesh);
  ASSERT_NE(nullptr, this->Scene->face_animation_group);
  ASSERT_NE(nullptr, this->Scene->mesh_material);
  ASSERT_NE(nullptr, this->Scene->base_faces);
}

TYPED_TEST(ImportedAnimatedScene, SetFrameTest)
{  
  ASSERT_EQ(SCENE_ENGINE_OK, system_finalize_scene(this->SceneSystem, this->Scene));

  int r = 0;
  r = scene_set_frame(this->Scene, 0);
  ASSERT_EQ(SCENE_ENGINE_OK, r);
}

template <int SceneEngineType, int EmissionType, int RadianceType, int RayCasterType>
struct AnimatedPlanesConfig
{
  static const int SceneSystem = SceneEngineType;
  static const int EmissionSystem = EmissionType;
  static const int RadianceSystem = RadianceType;
  static const int RayCasterSystem = RayCasterType;
};

template <typename ConfigT>
class AnimatedPlanes : public Test
{
public:
  typedef ConfigT Config;

  AnimatedPlanes()
  {
    SceneSystem = subject::system_create(ConfigT::SceneSystem);
    StaticScene = MakeAssymetric3PlanesScene(SceneSystem);
    AnimatedScene = MakeAssymetric3PlanesScene(SceneSystem);

    auto one_side_black_body = subject::black_body();
    one_side_black_body.front.emission.emissivity = 0;

    AddMaterial(StaticScene, subject::black_body());
    AddMaterial(StaticScene, one_side_black_body);
    AddMaterial(AnimatedScene, subject::black_body());
    AddMaterial(AnimatedScene, one_side_black_body);
    StaticScene->meshes[0].material_idx = 0;
    StaticScene->meshes[1].material_idx = 1;
    StaticScene->meshes[2].material_idx = 1;
    AnimatedScene->meshes[0].material_idx = 0;
    AnimatedScene->meshes[1].material_idx = 1;
    AnimatedScene->meshes[2].material_idx = 1;

    AnimatedScene->animation = (subject::animation_t*) calloc(1, sizeof(subject::animation_t));
    subject::animation_t* anim = AnimatedScene->animation;
    anim->n_groups = 1;
    anim->n_frames = 3;
    anim->n_static_faces = 2;
    anim->n_dynamic_faces = 4;
    anim->n_current_frame = 0;
    anim->frame_time_step = 1;
    anim->groups = (subject::face_group_t*)calloc(1, sizeof(subject::face_group_t));
    anim->groups[0].n_offset = 2;
    anim->groups[0].n_length = 4;
    anim->translations = (math::vec3*)calloc(3, sizeof(math::vec3));
    anim->rotations = (math::mat33*)calloc(3, sizeof(math::mat33));
    anim->rotations[0] = math::IDENTITY_33();
    anim->rotations[1] = math::rotate_towards(math::make_vec3(0, 0, 1), math::make_vec3(0, 0, -1));
    anim->rotations[2] = math::rotate_towards(math::make_vec3(0, 0, 1), math::normalize(math::make_vec3(-1, 0, 1)));

    subject::system_finalize_scene(SceneSystem, StaticScene);
    subject::system_finalize_scene(SceneSystem, AnimatedScene);
  }

  ~AnimatedPlanes()
  {
    scene_free(StaticScene);
    scene_free(AnimatedScene);
    system_free(SceneSystem);
  }

public:
  subject::system_t* SceneSystem;
  subject::scene_t* StaticScene;
  subject::scene_t* AnimatedScene;
};

template <typename ConfigT>
class AnimatedPlanesMalleyTest : public AnimatedPlanes<ConfigT>
{};

template <typename ConfigT>
class AnimatedPlanesParallelEmissionTest : public AnimatedPlanes<ConfigT>
{};

/**
  @detail All this mumgling because Optix ray caster does not support only-host scenes.
  @todo Support host scenes in Optix ray caster.
*/
typedef ::testing::Types<AnimatedPlanesConfig<SCENE_ENGINE_HOST, EMISSION_MALLEY_CPU, THERMAL_EQUATION_RADIANCE_CPU, RAY_CASTER_OPTIX>
> AnimatedPlanesConfigs;

typedef ::testing::Types<
  AnimatedPlanesConfig<SCENE_ENGINE_CUDA, EMISSION_MALLEY_CPU, THERMAL_EQUATION_RADIANCE_CPU, RAY_CASTER_OPTIX>,
  AnimatedPlanesConfig<SCENE_ENGINE_CUDA, EMISSION_MALLEY_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, RAY_CASTER_OPTIX>
> AnimatedPlanesMalleyConfigs;

typedef ::testing::Types<
  AnimatedPlanesConfig<SCENE_ENGINE_CUDA, EMISSION_PARALLEL_RAYS_CPU, THERMAL_EQUATION_RADIANCE_CPU, RAY_CASTER_OPTIX>,
  AnimatedPlanesConfig<SCENE_ENGINE_CUDA, EMISSION_PARALLEL_RAYS_CUDA, THERMAL_EQUATION_RADIANCE_CUDA, RAY_CASTER_OPTIX>
> AnimatedPlanesParallelEmissionConfigs;

TYPED_TEST_CASE(AnimatedPlanes, AnimatedPlanesConfigs);
TYPED_TEST_CASE(AnimatedPlanesMalleyTest, AnimatedPlanesMalleyConfigs);
TYPED_TEST_CASE(AnimatedPlanesParallelEmissionTest, AnimatedPlanesParallelEmissionConfigs);

TYPED_TEST(AnimatedPlanes, EqualBeforeRotation)
{
  subject::scene_relocate(this->StaticScene, -1);
  subject::scene_relocate(this->AnimatedScene, -1);

  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 0, 0));
  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 1, 1));
  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 2, 2));
}

TYPED_TEST(AnimatedPlanes, EqualAfterRotation)
{
  subject::scene_set_frame(this->AnimatedScene, 1);

  subject::scene_relocate(this->StaticScene, -1);
  subject::scene_relocate(this->AnimatedScene, -1);

  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 0, 0));
  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 1, 2));
  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 2, 1));
}

TYPED_TEST(AnimatedPlanesMalleyTest, AbsorptionTest)
{
  malley_cpu::params_t emitter_params;
  emitter_params.n_rays = 40000;
  auto emitter = emission::system_create(Config::EmissionSystem, &emitter_params);
  auto ray_caster = ray_caster::system_create(Config::RayCasterSystem);

  radiance_equation::params_t radiance_params = {};
  radiance_params.emitters = &emitter;
  radiance_params.n_emitters = 1;
  radiance_params.ray_caster = ray_caster;  
  auto radiance = thermal_equation::system_create(Config::RadianceSystem, &radiance_params);
  
  auto static_task = thermal_equation::task_create(this->StaticScene);
  auto dynamic_task = thermal_equation::task_create(this->AnimatedScene);

  float* mesh_temperatures = (float*)calloc(3, sizeof(float));
  mesh_temperatures[0] = 1000;
  mesh_temperatures[1] = 20;
  mesh_temperatures[2] = 20;
  static_task->temperatures = mesh_temperatures;
  dynamic_task->temperatures = mesh_temperatures;

  ASSERT_EQ(THERMAL_EQUATION_OK, system_set_scene(radiance, this->StaticScene));
  ASSERT_EQ(THERMAL_EQUATION_OK, system_calculate(radiance, static_task));

  ASSERT_EQ(THERMAL_EQUATION_OK, system_set_scene(radiance, this->AnimatedScene));

  // Check first frame equal to static scene
  ASSERT_EQ(THERMAL_EQUATION_OK, system_calculate(radiance, dynamic_task));
  ASSERT_NEAR(static_task->absorption[0], dynamic_task->absorption[0], 0.001); // small and noisy value
  ASSERT_NEAR(diff(static_task->absorption[1], dynamic_task->absorption[1]), 0, 0.001);
  ASSERT_NEAR(diff(static_task->absorption[2], dynamic_task->absorption[2]), 0, 0.001);

  // Update to first frame
  memset(dynamic_task->emission, 0, 3 * sizeof(float));
  memset(dynamic_task->absorption, 0, 3 * sizeof(float));
  subject::scene_set_frame(this->AnimatedScene, 1);
  ray_caster::system_update(ray_caster);
  ASSERT_EQ(THERMAL_EQUATION_OK, system_calculate(radiance, dynamic_task));

  // Sanity check
  ASSERT_NEAR(diff(static_task->emission[0], dynamic_task->emission[0]), 0, 0.001);
  ASSERT_NEAR(diff(static_task->emission[1], dynamic_task->emission[1]), 0, 0.001);
  ASSERT_NEAR(diff(static_task->emission[2], dynamic_task->emission[2]), 0, 0.0001);

  ASSERT_NEAR(static_task->absorption[0], dynamic_task->absorption[0], 0.001); // small and noisy value
  ASSERT_NEAR(diff(static_task->absorption[1], dynamic_task->absorption[2]), 0, 0.001);
  ASSERT_NEAR(diff(static_task->absorption[2], dynamic_task->absorption[1]), 0, 0.001);

  free(mesh_temperatures);
  task_free(static_task);
  task_free(dynamic_task);
  system_free(radiance);
  system_free(ray_caster);
  system_free(emitter);
}

TYPED_TEST(AnimatedPlanesParallelEmissionTest, AbsorptionTest)
{
  parallel_rays_emission_cpu::parallel_rays_source_t source_params;
  source_params.direction = math::make_vec3(0, 0, -1);
  source_params.n_rays = 15000;
  source_params.power = 1;

  parallel_rays_emission_cpu::params_t emitter_params;
  emitter_params.n_sources = 1;
  emitter_params.sources = &source_params;
  auto emitter = emission::system_create(Config::EmissionSystem, &emitter_params);
  auto ray_caster = ray_caster::system_create(Config::RayCasterSystem);

  radiance_equation::params_t radiance_params = {};
  radiance_params.emitters = &emitter;
  radiance_params.n_emitters = 1;
  radiance_params.ray_caster = ray_caster;
  auto radiance = thermal_equation::system_create(Config::RadianceSystem, &radiance_params);

  auto dynamic_task = thermal_equation::task_create(this->AnimatedScene);

  float mesh_temperatures[] = { 300, 300, 300 };
  mesh_temperatures[0] = 300;
  mesh_temperatures[1] = 300;
  mesh_temperatures[2] = 300;
  
  ASSERT_EQ(THERMAL_EQUATION_OK, system_set_scene(radiance, this->AnimatedScene));

  // Check first frame equal to static scene
  ASSERT_EQ(THERMAL_EQUATION_OK, system_calculate(radiance, dynamic_task));
  ASSERT_NEAR(0, dynamic_task->absorption[0], 0.01); // small and noisy value
  ASSERT_NEAR(1, dynamic_task->absorption[1], 0.02);
  ASSERT_NEAR(0, dynamic_task->absorption[2], 0.01);

  // Update to first frame
  memset(dynamic_task->emission, 0, 3 * sizeof(float));
  memset(dynamic_task->absorption, 0, 3 * sizeof(float));
  subject::scene_set_frame(this->AnimatedScene, 1);
  ray_caster::system_update(ray_caster);
  ASSERT_EQ(THERMAL_EQUATION_OK, system_calculate(radiance, dynamic_task));

  ASSERT_NEAR(0, dynamic_task->absorption[0], 0.01);
  ASSERT_NEAR(0, dynamic_task->absorption[1], 0.01);
  ASSERT_NEAR(1, dynamic_task->absorption[2], 0.02);

  memset(dynamic_task->emission, 0, 3 * sizeof(float));
  memset(dynamic_task->absorption, 0, 3 * sizeof(float));
  subject::scene_set_frame(this->AnimatedScene, 2);
  ray_caster::system_update(ray_caster);
  ASSERT_EQ(THERMAL_EQUATION_OK, system_calculate(radiance, dynamic_task));

  ASSERT_NEAR(0, dynamic_task->absorption[0], 0.01);
  ASSERT_NEAR(cosf(float(M_PI/4)), dynamic_task->absorption[1], 0.02);

  task_free(dynamic_task);
  system_free(radiance);
  system_free(ray_caster);
  system_free(emitter);
}
