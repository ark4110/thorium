// Copyright (c) 2015-2017 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Thorium is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with thorium.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "../subject/system.h"

namespace cuda_scene_engine
{
  // @todo Query for all device info.
  // @todo Extract hardware parameters as single instance (abstraction) and reuse in all cuda engines.
  struct system_t : subject::system_t
  {
    int set_frame_block_size;
  };

  /**
    @detail Scene relocate()/free_data() are hardware agnostic. So scene can perform alloc()/memcpy() on its own.
    @detail Scene set_frame() is hardware specific, so it must be performed using system api.
  */
  struct scene_t : subject::scene_t
  {
    system_t* system;
    int n_current_device;

    subject::scene_t* host_scene;

    int n_devices;
    subject::scene_t** device_scenes;
  };

  subject::system_t* system_create();
  int system_set_frame(system_t* system, scene_t* scene, int n_frame);

  void cuda_free_scene_data(subject::scene_t* scene);
  int cuda_relocate_scene(subject::scene_t* scene, int device);
  int cuda_set_frame(subject::scene_t* scene, int n_frame);
}
