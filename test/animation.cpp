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
#include "../subject/system.h"
#include "../math/mat.h"
#include "../subject/operations.h"

using namespace testing;

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

template <int SceneEngineType, int EmissionType>
struct AnimatedPlanesConfig
{
  static const int SceneSystem = SceneEngineType;
  static const int EmissionSystem = EmissionType;
};

template <typename Config>
class AnimatedPlanes : public Test
{
public:
  AnimatedPlanes()
  {
    SceneSystem = subject::system_create(Config::SceneSystem);
    StaticScene = MakeAssymetric3PlanesScene(SceneSystem);
    AnimatedScene = MakeAssymetric3PlanesScene(SceneSystem);

    AddMaterial(StaticScene, subject::black_body());
    StaticScene->meshes[0].material_idx = 0;
    StaticScene->meshes[1].material_idx = 0;
    StaticScene->meshes[1].material_idx = 0;

    AddMaterial(AnimatedScene, subject::black_body());
    AnimatedScene->meshes[0].material_idx = 0;
    AnimatedScene->meshes[1].material_idx = 0;
    AnimatedScene->meshes[1].material_idx = 0;

    AnimatedScene->animation = (subject::animation_t*) calloc(1, sizeof(subject::animation_t));
    subject::animation_t* anim = AnimatedScene->animation;
    anim->n_groups = 1;
    anim->n_frames = 2;
    anim->n_static_faces = 2;
    anim->n_dynamic_faces = 4;
    anim->n_current_frame = 0;
    anim->frame_time_step = 1;
    anim->groups = (subject::face_group_t*)calloc(1, sizeof(subject::face_group_t));
    anim->groups[0].n_offset = 2;
    anim->groups[0].n_length = 4;
    anim->translations = (math::vec3*)calloc(2, sizeof(math::vec3));
    anim->rotations = (math::mat33*)calloc(2, sizeof(math::mat33));
    anim->rotations[0] = math::IDENTITY_33();
    anim->rotations[1] = math::rotate_towards(math::make_vec3(0, 0, 1), math::make_vec3(0, 0, -1));

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

typedef ::testing::Types<AnimatedPlanesConfig<SCENE_ENGINE_HOST, EMISSION_MALLEY_CPU>, AnimatedPlanesConfig<SCENE_ENGINE_CUDA, EMISSION_MALLEY_CUDA>> AnimatedPlanesConfigs;

TYPED_TEST_CASE(AnimatedPlanes, AnimatedPlanesConfigs);

TYPED_TEST(AnimatedPlanes, EqualAfterRotation)
{
  subject::scene_set_frame(this->AnimatedScene, 1);

  subject::scene_relocate(this->StaticScene, -1);
  subject::scene_relocate(this->AnimatedScene, -1);

  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 0, 0));
  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 1, 2));
  ASSERT_TRUE(scenes_mesh_faces_near_enough(this->AnimatedScene, this->StaticScene, 2, 1));
}
