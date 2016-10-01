// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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

/**
 * This module contains performance test console utility for calculator (CPU) and ray caster (CPU, GPU).
 */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <cmath>

#include "../ray_caster/system.h"
#include "../form_factors/system.h"
#include "../math/grid.h"
#include "../math/operations.h"
#include "../emission/malley_cpu.h"
#include "../thermal_solution/system.h"
#include "../thermal_equation/system.h"
#include "../thermal_equation/radiance_cpu.h"
#include "../subject/generator.h"
#include "../import_export/obj_export.h"
#include "../import_export/obj_import.h"

#include <helper_timer.h>

using namespace ray_caster;

/// @brief Generates random triangles with vertices located on sphere's surface (confette scene).
scene_t* MakeConfettiScene(int n_faces, float radius, subject::generator_t* generator)
{
  face_t* faces = (face_t*)malloc(n_faces * sizeof(face_t));
  scene_t* scene = (scene_t*)malloc(sizeof(scene_t));
  *scene = { n_faces, faces };
  for (int i = 0; i != n_faces; ++i)
  {
    face_t& f = faces[i];
    generator_surface_point(generator, 3, f.points);
    math::vec3 position;
    generator_volume_point(generator, 1, &position);
    position *= radius;
    f.points[0] += position;
    f.points[1] += position;
    f.points[2] += position;
  }  
  
  return scene;
}

/// @brief Creates task with n_rays rays from sphere's surface towards center.
task_t* MakeCollapsingRays(int n_rays, float radius, subject::generator_t* generator)
{
  math::ray_t* rays = (math::ray_t*)malloc(n_rays * sizeof(math::ray_t));
  if (rays == 0)
    throw std::out_of_range("not enough memory to allocate rays");
  face_t** hit_face = (face_t**)malloc(n_rays * sizeof(face_t*));
  math::vec3* hit_point = (math::vec3*)malloc(n_rays * sizeof(math::vec3));
  task_t* task = (task_t*)malloc(sizeof(task_t));
  *task = { n_rays, rays, hit_face, hit_point };
  for (int i = 0; i != n_rays; ++i)
  { 
    math::vec3 direction;
    generator_surface_point(generator, 1, &direction);
    direction *= (2 + radius);
    // Origin is scaled by 10% to direction - rays directed towards zero.
    math::vec3 origin = direction * 1.1f;
    rays[i] = { origin, direction - origin };
  }

  return task;
}

/// @brief Clones given task. Uses to copy task from GPU to CPU.
task_t* task_clone(task_t* task)
{
  task_t* result = (task_t*)malloc(sizeof(task_t));
  result->n_tasks = task->n_tasks;
  result->ray = task->ray;
  result->hit_face = (face_t**)malloc(task->n_tasks * sizeof(face_t*));
  result->hit_point = (math::vec3*)malloc(task->n_tasks * sizeof(math::vec3));
  return result;
}

/// @brief Error struct for @see CalculateError(task_t* cpu, task_t* gpu).
struct ErrorStats
{
  ErrorStats()
  : TotalDistance(0)
  , MaxDistance(0)
  , AverageDistance(0)
  , Mismatch(0)
  {
  }

  float TotalDistance;
  float MaxDistance;
  float AverageDistance;
  int Mismatch;
};

/// @brief Checks errors between tasks in different ray caster implementations.
ErrorStats CalculateError(task_t* cpu, task_t* gpu)
{
  ErrorStats result = {};
  for (int i = 0; i != cpu->n_tasks; ++i)
  {
    if (cpu->hit_face[i] != gpu->hit_face[i])
    {
      ++result.Mismatch;
      continue;
    }

    if (cpu->hit_face[i] != 0)
    {
      math::vec3 diff = cpu->hit_point[i] - gpu->hit_point[i];
      float distance = sqrtf(dot(diff, diff));
      result.TotalDistance += distance;
      if (distance > result.MaxDistance)
        result.MaxDistance = distance;
    }
  }

  result.AverageDistance = result.TotalDistance / cpu->n_tasks;

  return result;
}

/// @brief Checks for non-equal ray intersections between tasks in different ray caster implementations.
int CalculateMismatch(task_t* cpu, task_t* gpu)
{
  int result = 0;
  for (int i = 0; i != cpu->n_tasks; ++i)
  {
    if (cpu->hit_face[i] != gpu->hit_face[i])
    {
      ++result;
    }
  }

  return result;
}

#include <functional>

int align(int pos, int column, const int* columns)
{
  char buf[32];
  memset(buf, ' ', sizeof(buf));
  buf[columns[column] - pos] = 0;
  printf(buf);
  return columns[column];
}

template <typename T>
void printErrorStatsTable(const int* activeSystems, T stats, const char* taskName, std::function<void(T stats, int i, int j, char* buf)> fieldPrinter)
{
  int columns[RAY_CASTER_SYSTEMS_COUNT] = {};
  for (int s = 0; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
    if (activeSystems[s])
      if (strlen(ray_caster::system_get_id(s)) + 1 > columns[0])
        columns[0] = strlen(ray_caster::system_get_id(s)) + 1;

  int prev = columns[0];
  for (int s = 1; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
  {
    if (activeSystems[s])
    {
      columns[s] = prev;
      prev += strlen(ray_caster::system_get_id(s)) + 1;
    }
  }

  char buf[256];
  printf("%s\n", taskName);
  int pos = 0;
  for (int j = 0; j != RAY_CASTER_SYSTEMS_COUNT; ++j)
    if (activeSystems[j])
    {
      pos = align(pos, j, columns);
      const char* name = ray_caster::system_get_id(j);
      printf("%s", name);
      pos += strlen(name);

    }
  printf("\n");

  for (int i = 0; i != RAY_CASTER_SYSTEMS_COUNT; ++i)
  {
    if (!activeSystems[i])
      continue;

    const char* name = ray_caster::system_get_id(i);
    printf("%s", name);
    int pos = strlen(name);
    for (int j = 0; j != RAY_CASTER_SYSTEMS_COUNT; ++j)
    {
      if (!activeSystems[j])
        continue;

      pos = align(pos, j, columns);

      if (i == j)
      {
        printf("-");
        pos += 1;
        continue;
      }

      fieldPrinter(stats, i, j, buf);
      printf("%s", buf);
      pos += strlen(buf);
    }
    printf("\n");
  }
}

int main(int argc, char* argv[])
{
  if (argc == 1)
  {
    printf("Usage: perfTest [n_faces] [engines]*\n");
    return -1;
  }

  system_t* systems[RAY_CASTER_SYSTEMS_COUNT];
  for (int s = 0; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
    systems[s] = system_create(s);

  int systemsToCheck[RAY_CASTER_SYSTEMS_COUNT] = {};

  if (argc > 2)
  {
    for (int a = 2; a != argc; ++a)
    {
      for (int s = 0; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
        if (strcmp(ray_caster::system_get_id(s), argv[a]) == 0)
          systemsToCheck[s] = 1;
    }
  }
  else
  {
    systemsToCheck[RAY_CASTER_ZGRID_CUDA] = 1;
    systemsToCheck[RAY_CASTER_OPTIX] = 1; 
  }

  task_t* tasks[RAY_CASTER_SYSTEMS_COUNT] = {};

  try
  {
    int n_faces = atoi(argv[1]);
    int n_rays = 100 * n_faces;
    bool no_cpu = false;
    bool no_radiance = false;

    float radius = pow(n_faces, 0.3);
    
    subject::generator_t* generator = subject::generator_create_spherical();
    // Create random scene for ray caster.
    scene_t* scene = 0;
    ray_caster::scene_t* rs = 0;
    subject::scene_t* ss = 0;

    printf("Generating confetti scene with %d elements...\n", n_faces);
    rs = MakeConfettiScene(n_faces, radius, generator);
    scene = rs;

    /*{
      printf("Loading scene from '%s'...\n", argv[1]);
      if (int r = obj_import::scene(argv[1], &ss))
        return r;
      n_rays = 100 * ss->n_faces;
      scene = (scene_t*)ss;
    }*/
    
    if (systemsToCheck[RAY_CASTER_ZGRID_CPU] || systemsToCheck[RAY_CASTER_ZGRID_CUDA])
      math::grid_draw_hist(1024, scene->faces, scene->n_faces);
    
    printf("Generating %d collapsing rays...\n", n_rays);
    task_t* baseTask = MakeCollapsingRays(n_rays, radius, generator);    
    for(int s = 0; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
      if (systemsToCheck[s])
        tasks[s] = task_clone(baseTask);

    double timings[RAY_CASTER_SYSTEMS_COUNT] = {};
    
    StopWatchInterface *hTimer;
    sdkCreateTimer(&hTimer);
    for (int s = 0; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
    {
      if (!systemsToCheck[s])
      {
        printf("Skip %s system test\n", ray_caster::system_get_name(s));
        timings[s] = -1;
        continue;
      }
      sdkResetTimer(&hTimer);
      sdkStartTimer(&hTimer);

      printf("Casting scene on %s ...\n", ray_caster::system_get_name(s));
      ray_caster::system_t* system = systems[s];
      system_set_scene(system, scene);
      system_prepare(system);
      system_cast(system, tasks[s]);
      sdkStopTimer(&hTimer);
      timings[s] = 1.0e-3 * sdkGetTimerValue(&hTimer);
      printf("Done in %fs.\n", timings[s]);
    }

    ErrorStats stats[RAY_CASTER_SYSTEMS_COUNT][RAY_CASTER_SYSTEMS_COUNT] = {};
    for (int i = 0; i != RAY_CASTER_SYSTEMS_COUNT; ++i)
    {
      if (!systemsToCheck[i])
        continue;
      for (int j = i; j != RAY_CASTER_SYSTEMS_COUNT; ++j)
      {
        if (!systemsToCheck[j])
          continue;
        if (i == j)
          continue;
        stats[i][j] = CalculateError(tasks[i], tasks[j]);
        stats[j][i] = stats[i][j];
      }
    }

    printErrorStatsTable<ErrorStats*>(systemsToCheck, (ErrorStats*)stats, "\nAverage diff distances:", [](ErrorStats* stats, int i, int j, char* buf) {
      sprintf(buf, "%.3Ef", stats[i * RAY_CASTER_SYSTEMS_COUNT + j].AverageDistance);
    });

    printErrorStatsTable<ErrorStats*>(systemsToCheck, (ErrorStats*)stats, "\nMax diff distances:", [](ErrorStats* stats, int i, int j, char* buf) {
      sprintf(buf, "%.3Ef", stats[i * RAY_CASTER_SYSTEMS_COUNT + j].MaxDistance);
    });

    printErrorStatsTable<ErrorStats*>(systemsToCheck, (ErrorStats*)stats, "\nTotal diff distances:", [](ErrorStats* stats, int i, int j, char* buf) {
      sprintf(buf, "%.3Ef", stats[i * RAY_CASTER_SYSTEMS_COUNT + j].TotalDistance);
    });

    printErrorStatsTable<ErrorStats*>(systemsToCheck, (ErrorStats*)stats, "\nHit mismatch counts:", [](ErrorStats* stats, int i, int j, char* buf) {
      sprintf(buf, "%d", stats[i * RAY_CASTER_SYSTEMS_COUNT + j].Mismatch);
    });

    printErrorStatsTable<double*>(systemsToCheck, timings, "\nTimings:", [](double* timings, int i, int j, char* buf) {
      sprintf(buf, "%f", timings[i] / timings[j]);
    }); 

    scene_free(rs);
    scene_free(ss);

    free(baseTask->ray);
    for (int s = 0; s != RAY_CASTER_SYSTEMS_COUNT; ++s)
    {
      if (systemsToCheck[s])
      {
        free(tasks[s]->hit_face);
        free(tasks[s]->hit_point);
        free(tasks[s]);
      }
    }

    generator_free(generator);

    return 0;
  }
  catch (const std::exception& e)
  {
    printf("%s\n", e.what());
    return -1;
  }
}

