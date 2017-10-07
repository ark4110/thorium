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
// along with thorium.  If not, see <http://www.gnu.org/licenses/>.

#include "gtest/gtest.h"
#include "test_scene.h"
#include "ray_caster/system.h"
#include "math/operations.h"
#include "math/triangle.h"

using namespace ray_caster;
using math::vec3;
using math::make_vec3;
using namespace testing;

template <int SceneEngineTypeID, int RayCasterEngineTypeID>
struct EngineConfig
{
  static const int SceneID = SceneEngineTypeID;
  static const int RayCasterID = RayCasterEngineTypeID;
};

template <typename EngineConfig>
class RayCaster : public Test
{
public:
  RayCaster()
  {
    SceneSystem = subject::system_create(EngineConfig::SceneID);
    System = system_create(EngineConfig::RayCasterID);
  }

  ~RayCaster()
  {
    system_free(this->System);
    system_free(this->SceneSystem);
  }
 
  subject::system_t* SceneSystem;
  system_t* System;
};

typedef ::testing::Types<EngineConfig<SCENE_ENGINE_HOST, RAY_CASTER_NAIVE_CPU>, EngineConfig<SCENE_ENGINE_CUDA, RAY_CASTER_NAIVE_CPU>, EngineConfig<SCENE_ENGINE_CUDA, RAY_CASTER_NAIVE_CUDA>
#if defined _WIN64 || !defined  _WIN32
  , EngineConfig<SCENE_ENGINE_CUDA, RAY_CASTER_OPTIX>
#endif
> RayCasterTypes;
TYPED_TEST_CASE(RayCaster, RayCasterTypes);

TYPED_TEST(RayCaster, MemoryManagementIsCorrect)
{
  // @todo how to check?
  // Not crashing is ok already.
}

TYPED_TEST(RayCaster, AcceptEmptyScene)
{
  subject::scene_t* emptyScence = MakeEmptyScene(this->SceneSystem);
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(this->System, emptyScence));
  scene_free(emptyScence);
}

TYPED_TEST(RayCaster, AcceptFloorScene)
{
  subject::scene_t* floorScene = MakeFloorScene(this->SceneSystem);
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(this->System, floorScene));
  scene_free(floorScene);
}

TYPED_TEST(RayCaster, PrepareFailsForNoScene)
{ 
  ASSERT_EQ(-RAY_CASTER_ERROR, system_update(this->System));
}

TYPED_TEST(RayCaster, AcceptConsequencesSetScene)
{
  subject::scene_t* scene1 = MakeFloorScene(this->SceneSystem);
  subject::scene_t* scene2 = MakeFloorScene(this->SceneSystem);
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(this->System, scene1));
  ASSERT_EQ(RAY_CASTER_OK, system_set_scene(this->System, scene2));
  scene_free(scene1);
  scene_free(scene2);
}

TYPED_TEST(RayCaster, PrepareFailsForEmptyScene)
{
  subject::scene_t* emptyScence = MakeEmptyScene(this->SceneSystem);
  system_set_scene(this->System, emptyScence);
  ASSERT_EQ(-RAY_CASTER_ERROR, system_update(this->System));
  scene_free(emptyScence);
}

TYPED_TEST(RayCaster, PreparePassForNotEmptyScene)
{
  subject::scene_t* floorScene = MakeFloorScene(this->SceneSystem);
  system_set_scene(this->System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));
  scene_free(floorScene);
}

TYPED_TEST(RayCaster, ProcessAllRays)
{
  subject::scene_t* floorScene = MakeFloorScene(this->SceneSystem);

  system_set_scene(this->System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));

  task_t* task = task_create(2);

  vec3 center0 = triangle_center(floorScene->faces[0]);
  vec3 origin0 = center0 + make_vec3(0.f, 0.f, 1.f);
  task->ray[0] = ray_to_triangle(origin0, floorScene->faces[0]);

  vec3 center1 = triangle_center(floorScene->faces[1]);
  vec3 origin1 = center1 + make_vec3(0.f, 0.f, -1.f);
  task->ray[1] = ray_to_triangle(origin1, floorScene->faces[1]);

  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center0));
  ASSERT_TRUE(near_enough(task_hit_point(task, 1), center1));

  scene_free(floorScene);
  task_free(task);
}

TYPED_TEST(RayCaster, JustWorks)
{
  subject::scene_t* floorScene = MakeFloorScene(this->SceneSystem);

  system_set_scene(this->System, floorScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));
  
  task_t* task = task_create(1);

  vec3 center = triangle_center(floorScene->faces[0]);
  vec3 origin = center + make_vec3(0.f, 0.f, 1.f);
  task->ray[0] = ray_to_triangle(origin, floorScene->faces[0]);
  
  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center));

  scene_free(floorScene);
  task_free(task);
}

TYPED_TEST(RayCaster, HandleRayIntersectingTriangle)
{
  subject::scene_t* stackScene = MakeStackScene(this->SceneSystem);

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));

  task_t* task = task_create(1);

  vec3 center = triangle_center(stackScene->faces[0]);
  vec3 origin = center + make_vec3(0.f, 0.f, .5f);
  vec3 direction = make_vec3(0.f, 0.f, -.5f);
  task->ray[0] = { origin, direction };
  
  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center));

  scene_free(stackScene);
  task_free(task);
}

TYPED_TEST(RayCaster, FindNearestTriangle)
{
  subject::scene_t* stackScene = MakeStackScene(this->SceneSystem);

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));

  task_t* task = task_create(1);

  vec3 center = triangle_center(stackScene->faces[0]);
  vec3 origin = center + make_vec3(0.f, 0.f, 1.f);
  task->ray[0] = ray_to_triangle(origin, stackScene->faces[0]);
  
  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center));

  scene_free(stackScene);
  task_free(task);
}

TYPED_TEST(RayCaster, SkipTriangleNearButInOppositeDirection)
{
  subject::scene_t* stackScene = MakeStackScene(this->SceneSystem);

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));

  task_t* task = task_create(1);

  vec3 center = triangle_center(stackScene->faces[1]);
  // origin near to first stack triangle but ray is in opposite direction
  vec3 origin = center + make_vec3(0.f, 0.f, .9f);
  task->ray[0] = ray_to_triangle(origin, stackScene->faces[1]);
  
  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center));

  scene_free(stackScene);
  task_free(task);
}

TYPED_TEST(RayCaster, IntersectTriganglesNotPlanes)
{
  subject::scene_t* stackScene = MakeStackScene(this->SceneSystem);

  system_set_scene(this->System, stackScene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));

  task_t* task = task_create(1);

  vec3 center = triangle_center(stackScene->faces[1]);
  // Ray hit first triangle plane but bypass it by long side
  vec3 origin = center + make_vec3(2.f, 2.f, 2.f);
  task->ray[0] = ray_to_triangle(origin, stackScene->faces[1]);

  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center));

  scene_free(stackScene);
  task_free(task);
}

TYPED_TEST(RayCaster, ProcessLargeScene)
{
  subject::scene_t* scene = 0;
  ASSERT_EQ(SCENE_ENGINE_OK, system_create_scene(this->SceneSystem, &scene));
  scene->n_faces = 1000;
  scene->faces = (face_t*)malloc(sizeof(face_t)* scene->n_faces);

  system_finalize_scene(this->SceneSystem, scene);

  face_t reference = make_floor_face1();
  for (int i = -scene->n_faces / 2; i != scene->n_faces / 2; ++i)
  {
    face_t face = reference;
    vec3 shift = { 0.f, 0.f, .1f * (float)i };
    face.points[0] += shift;
    face.points[1] += shift;
    face.points[2] += shift;
    scene->faces[i + scene->n_faces / 2] = face;
  }

  system_set_scene(this->System, scene);
  ASSERT_EQ(RAY_CASTER_OK, system_update(this->System));

  task_t* task = task_create(1);

  vec3 center = triangle_center(scene->faces[0]);
  // Ray hit first triangle plane but bypass it by long side
  vec3 origin = center + make_vec3(2.f, 2.f, -2.f);
  task->ray[0] = ray_to_triangle(origin, scene->faces[0]);
  
  system_relocate_task(this->System, task, 0);
  ASSERT_EQ(RAY_CASTER_OK, system_cast(this->System, task));
  system_relocate_task(this->System, task, -1);

  // @todo Provide gtest comparison overloads for vec3
  ASSERT_TRUE(near_enough(task_hit_point(task, 0), center));

  task_free(task);
  scene_free(scene);
}
