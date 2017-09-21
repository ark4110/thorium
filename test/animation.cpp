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

#include "../import_export/obj_import.h"
#include "../subject/system.h"

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
class AnimatedScene : public Test
{
public:
  AnimatedScene()
  {
    SceneSystem = subject::system_create(SceneEngineType::ID);
    system_create_scene(SceneSystem, &Scene);
    obj_import::scene("models/test_hierarchy.obj", Scene);
  }

  ~AnimatedScene()
  {
    scene_free(Scene);
    system_free(SceneSystem);
  }

public:
  subject::system_t* SceneSystem;
  subject::scene_t* Scene;
};

typedef ::testing::Types<SceneEngineType<SCENE_ENGINE_HOST>, SceneEngineType<SCENE_ENGINE_CUDA>> SceneEnginesTypes;

TYPED_TEST_CASE(AnimatedScene, SceneEnginesTypes);

TYPED_TEST(AnimatedScene, FinalizationTest)
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

TYPED_TEST(AnimatedScene, SetFrameTest)
{  
  ASSERT_EQ(SCENE_ENGINE_OK, system_finalize_scene(this->SceneSystem, this->Scene));

  int r = 0;
  r = scene_set_frame(this->Scene, 0);
  ASSERT_EQ(SCENE_ENGINE_OK, r);
}
