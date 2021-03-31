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

#include "math/operations.h"
#include "math/triangle.h"
#include "math/tetrahedron.h"
#include "emission/system.h"
#include "form_factors/system.h"
#include "thermal_solution/system.h"
#include "thermal_equation/system.h"
#include "thermal_equation/conductive_cpu.h"
#include "subject/objects.h"

using namespace testing;

namespace
{
  class ConductiveSolution
          : public Test
  {
  public:
    ConductiveSolution()
    {
      conductive_equation::params_t params;
      Equation = thermal_equation::system_create(THERMAL_EQUATION_CONDUCTIVE_CPU, &params);

      SolutionParams = {1, &Equation};
      System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &SolutionParams);

      //Materials[0] = subject::black_body();
      Materials[0] = subject::material_Al(0.02f);

      Scene = {12, Faces,
               6, Meshes,
               1, Materials};

      memcpy(Faces, subject::box(), sizeof(subject::face_t) * 12);
      for (int m = 0; m != 6; ++m)
        Scene.meshes[m] = {2 * m, 2};

      memset(Temperatures, 0, sizeof(Temperatures));
      Temperatures[0] = 300.f;

      Task = thermal_solution::task_create(Scene.n_meshes);
      Task->time_delta = 1.f;
    }

    ~ConductiveSolution()
    {
      thermal_solution::task_free(Task);
      thermal_solution::system_free(System);
      thermal_equation::system_free(Equation);
    }

    float CalculateEnergy(float *temperatures)
    {
      float e = 0;
      for (int i = 0; i != Scene.n_meshes; ++i)
        e += temperatures[i];
      return e;
    }

    thermal_equation::system_t *Equation;
    thermal_solution::params_t SolutionParams;
    thermal_solution::system_t *System;
    subject::material_t Materials[1];
    subject::face_t Faces[12];
    subject::mesh_t Meshes[6];
    subject::scene_t Scene;
    float Temperatures[6];
    thermal_solution::task_t *Task;
  };
}

TEST_F(ConductiveSolution, BoxPreserveEnergy)
{
  using namespace thermal_solution;

  const float initialEnergy = CalculateEnergy(Temperatures);
  int r = 0;
  host_scene_finalize(&Scene);
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  for (int step = 1; step < 50; ++step)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);

  float currentEnergy = CalculateEnergy(Task->temperatures);
  ASSERT_NEAR(1.f, currentEnergy / initialEnergy, 0.001f);
}

TEST_F(ConductiveSolution, ReachBalance)
{
  using namespace thermal_solution;
  Task->time_delta = 30;

  const float initialEnergy = CalculateEnergy(Temperatures);
  int r = 0;
  host_scene_finalize(&Scene);
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  while (Task->n_step < 500 && fabsf(Task->temperatures[5] - Task->temperatures[0]) > 1)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  
  ASSERT_LE(Task->n_step, 500);
  float currentEnergy = CalculateEnergy(Task->temperatures);
  ASSERT_NEAR(1.f, currentEnergy / initialEnergy, 0.001f);
}

namespace
{
  class TetrahedronConductiveSolution
          : public Test
  {
  public:
    TetrahedronConductiveSolution()
    {
      conductive_equation::params_t params;
      Equation = thermal_equation::system_create(THERMAL_EQUATION_CONDUCTIVE_CPU, &params);

      SolutionParams = {1, &Equation};
      System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &SolutionParams);

      Materials[0] = subject::material_Al(0.02f);
      Materials[0].shell.thickness = 0;

      Scene = {8, Faces,
               2, Meshes,
               1, Materials};

      math::tetrahedron_t t0 = {
              {{0, 0, 0},
               {1, 0, 0},
               {0, 1, 0},
               {0, 0, 1}}};
      math::tetrahedron_t t1 = {
              {{0, 0, 0},
               {1, 0, 0},
               {0, -1, 0},
               {0, 0, 1}}};

      math::tetrahedron_build_faces(t0, Faces);
      math::tetrahedron_build_faces(t1, Faces + 4);

      Scene.meshes[0] = {0, 4};
      Scene.meshes[1] = {4, 4};

      Temperatures[0] = 290.f;
      Temperatures[1] = 310.f;

      Task = thermal_solution::task_create(Scene.n_meshes);
      Task->time_delta = 0.1f;
    }

    ~TetrahedronConductiveSolution()
    {
      thermal_solution::task_free(Task);
      thermal_solution::system_free(System);
      thermal_equation::system_free(Equation);
    }

    float CalculateEnergy(float *temperatures)
    {
      float e = 0;
      for (int i = 0; i != Scene.n_meshes; ++i)
        e += temperatures[i];
      return e;
    }

    thermal_equation::system_t *Equation;
    thermal_solution::params_t SolutionParams;
    thermal_solution::system_t *System;
    subject::material_t Materials[1];
    subject::face_t Faces[8];
    subject::mesh_t Meshes[2];
    subject::scene_t Scene;
    float Temperatures[2];
    thermal_solution::task_t *Task;
  };
}

TEST_F(TetrahedronConductiveSolution, ReachBalance)
{
  using namespace thermal_solution;
  Task->time_delta = 2;

  const float initialEnergy = CalculateEnergy(Temperatures);
  int r = 0;
  host_scene_finalize(&Scene);
  r = system_set_scene(System, &Scene, Temperatures);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  while (Task->n_step < 500 && fabsf(Task->temperatures[1] - Task->temperatures[0]) > 0.1)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }

  ASSERT_LE(Task->n_step, 500);
  float currentEnergy = CalculateEnergy(Task->temperatures);
  ASSERT_NEAR(1.f, currentEnergy / initialEnergy, 0.001f);
}

