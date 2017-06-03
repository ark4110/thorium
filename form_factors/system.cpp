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
 * This module contains basic types to represent a scene for form factors calculation.
 * Module also contains base type (system_t) for form factor calculation with table of virtual methods.
 */

#include "system.h"
#include "cpu_system.h"
#include <stdlib.h>

namespace form_factors
{ 
  system_t* system_create(int type, const params_t* params)
  {
    system_t* system = 0;
    switch (type)
    {
    case FORM_FACTORS_CPU:
      system = cpu_form_factors::system_create();
      break;

    default:
      return 0;
    }

    system_init(system, params);

    return system;
  }

  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  task_t* task_create(subject::scene_t* scene)
  {
    task_t* task = (task_t*)malloc(sizeof(task_t));
    task->form_factors = (float*)malloc(scene->n_meshes * scene->n_meshes * sizeof(float));
    return task;
  }

  void task_free(task_t* task)
  {
    free(task->form_factors);
    free(task);
  }

  int system_init(system_t* system, const params_t* params)
  {
    return system->methods->init(system, params);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, subject::scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_calculate(system_t* system, task_t* task)
  {
    return system->methods->calculate(system, task);
  }
}
