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

#include <math/rod.h>
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
      float step = 1;

      conductive_equation::params_t params = {};
      params.step = step;

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
      Task->time_delta = step;
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
      float step = 0.1;

      conductive_equation::params_t params = {};
      params.step = step;

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
      Task->time_delta = step;
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

namespace
{
  struct TetrahedronRod
  {
    TetrahedronRod(int count)
    {
      math::rod_t cube = subject::rod_cube();
      math::tetrahedron_t tetras[6];
      math::rod_to_six_tetrahedrons(cube, tetras);

      std::vector<math::face_t> faces(24);
      for (int i = 0; i != 6; ++i)
        math::tetrahedron_build_faces(tetras[i], faces.data() + i * 4);

      for (int i = 0; i != count; ++i)
      {
        auto current_faces = faces;
        bool invert_x = i % 2 != 0;
        float offset = 1.f * i;
        for (auto& f: current_faces)
        {
            for (auto& p: f.points)
              p.x = offset + invert_x ? (-p.x + 1) : p.x;
        }
        Faces.insert(Faces.end(), current_faces.begin(), current_faces.end());

        for (int mi = 0; mi != 6; ++mi)
        {
          subject::mesh_t mesh = {};
          mesh.first_idx = i * 6 * 4 + mi * 4;
          mesh.n_faces = 4;
          mesh.material_idx = 0;
          Meshes.push_back(mesh);
        }
      }
    }

    std::vector<subject::face_t> Faces;
    std::vector<subject::mesh_t> Meshes;
  };

  struct TetrahedronRodConductiveSolution
          : TetrahedronRod,
            public Test
  {
    static const int RodLength = 11;
    static constexpr float StartTemperature = 300;

    TetrahedronRodConductiveSolution()
            : TetrahedronRod(RodLength)
    {
      float step = 0.1;

      Materials[0] = subject::material_Al(0);
      Materials[0].front = {};
      Materials[0].rear = {};

      Scene = {(int) Faces.size(), Faces.data(),
               (int) Meshes.size(), Meshes.data(),
               1, Materials};

      conductive_equation::params_t params = {};
      params.step = step;
      Equation = thermal_equation::system_create(THERMAL_EQUATION_CONDUCTIVE_CPU, &params);

      SolutionParams = {1, &Equation};
      System = thermal_solution::system_create(THERMAL_SOLUTION_CPU_ADAMS, &SolutionParams);

      for (size_t mi = 0; mi != Meshes.size(); ++mi)
      {
        float element_index = mi / 6;
        Temperatures.push_back(StartTemperature + element_index);
      }

      Task = thermal_solution::task_create(Scene.n_meshes);
      Task->time_delta = step;
    }

    ~TetrahedronRodConductiveSolution()
    {
      thermal_solution::task_free(Task);
      thermal_solution::system_free(System);
      thermal_equation::system_free(Equation);
    }

    std::pair<float, float> CalculateEnergy(const float* temperatures) const
    {
      double volume_acc = 0;
      double energy_acc = 0;
      for (size_t mi = 0; mi != Meshes.size(); ++mi)
      {
        auto meshStat = CalculateMeshEnergy((int)mi, temperatures);
        volume_acc += meshStat.first;
        energy_acc += meshStat.second;
      }

      return {(float)volume_acc, (float)energy_acc};
    }

    std::pair<float, float> CalculateMeshEnergy(int mi, const float* temperatures) const
    {
      math::tetrahedron_t tetra;
      math::tetrahedron_build_from_faces(Faces.data() + 4 * mi, tetra.points);
      float volume = math::tetrahedron_volume(tetra);

      return {volume, volume * temperatures[mi]};
    }

    thermal_equation::system_t *Equation;
    thermal_solution::params_t SolutionParams;
    thermal_solution::system_t *System;
    subject::material_t Materials[1];
    subject::scene_t Scene;
    std::vector<float> Temperatures;
    thermal_solution::task_t *Task;
  };
}

TEST_F(TetrahedronRodConductiveSolution, ReachBalance)
{
  using namespace thermal_solution;
  Task->time_delta = 1;

  const auto initialState = CalculateEnergy(Temperatures.data());
  ASSERT_NEAR(RodLength, initialState.first, 1e-5);
  ASSERT_NEAR(StartTemperature * RodLength + RodLength * ((float)RodLength - 1) / 2, initialState.second, 1e-5);
  int r = 0;
  host_scene_finalize(&Scene);
  r = system_set_scene(System, &Scene, Temperatures.data());
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  r = system_calculate(System, Task);
  ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  while (Task->n_step < 50000 && fabsf(Task->temperatures[Meshes.size() - 1] - Task->temperatures[0]) > 0.01)
  {
    r = system_calculate(System, Task);
    ASSERT_EQ(THERMAL_SOLUTION_OK, r);
  }
  auto currentState = CalculateEnergy(Task->temperatures);
  ASSERT_NEAR(1.f, currentState.second / initialState.second, 0.001f);

  float avgEnergy = initialState.second / initialState.first;

  auto beginState = CalculateMeshEnergy(0, Task->temperatures);
  ASSERT_NEAR(beginState.second / beginState.first, avgEnergy, 0.001f);
}

