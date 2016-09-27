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
    malley_cpu::params_t params;
    params.n_rays = 1;
    Emitter = emission::system_create(EMISSION_MALLEY_CPU, &params);
    
    Scene = ToSubjectScene(MakeFloorScene());
    AddMaterial(Scene, subject::black_body());
    AddMesh(Scene, { 0, 2, 0 });

    system_set_scene(Emitter, Scene);
  }

  ~MalleyEmission()
  {
    system_free(Emitter);
    system_free(Caster);
    scene_free(Scene);
  }

  ray_caster::system_t* Caster;
  emission::system_t* Emitter;
  subject::scene_t* Scene;
  subject::material_t Material;
  subject::mesh_t Mesh;
};

TEST_F(MalleyEmission, AllocateRaycasterTask)
{ 
  auto task = emission::task_create();
  float b[2] = { 1000.f, 1000.f };
  task->mesh_emission = b;
  task->mesh_temperatures = b;
  system_calculate(Emitter, task);  
  ASSERT_EQ(4, task->rays->n_tasks);
}

TEST_F(MalleyEmission, ReallocateRaycasterTask)
{
  auto task = emission::task_create();
  float b[2] = { 1000.f, 1000.f };
  task->mesh_emission = b;
  task->mesh_temperatures = b;

  system_calculate(Emitter, task);
  system_calculate(Emitter, task);
  ASSERT_EQ(8, task->rays->n_tasks);
}
