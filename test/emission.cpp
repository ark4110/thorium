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
#include "emission/malley_emission.h"
#include "math/operations.h"
#include "math/triangle.h"


using math::vec3;
using math::make_vec3;
using namespace testing;

class MalleyEmission : public Test
{
public:
  MalleyEmission()
  {
    Caster = ray_caster::system_create(RAY_CASTER_NAIVE_CPU);
    Emitter = emission::system_create(EMISSION_MALLEY_CPU, Caster);
    Scene = MakeFloorScene();
    system_set_scene(Emitter, Scene);
  }

  ~MalleyEmission()
  {
    system_free(Emitter);
    system_free(Caster);
    ray_caster::scene_free(Scene);
  }

  ray_caster::system_t* Caster;
  emission::system_t* Emitter;
  ray_caster::scene_t* Scene;
};

TEST_F(MalleyEmission, AllocateRaycasterTask)
{ 
  auto task = malley_emission::task_create(1, 2);
  task->total_weight = 2.f;
  task->weights[0] = 1.f;
  task->weights[1] = 1.f;

  system_calculate(Emitter, task);  
  ASSERT_EQ(2, task->rays->n_tasks);
}

TEST_F(MalleyEmission, ReallocateRaycasterTask)
{
  auto task = malley_emission::task_create(1, 2);
  task->total_weight = 2.f;
  task->weights[0] = 1.f;
  task->weights[1] = 1.f;

  system_calculate(Emitter, task);
  system_calculate(Emitter, task);
  ASSERT_EQ(4, task->rays->n_tasks);
}