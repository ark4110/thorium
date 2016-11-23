// Copyright (c) 2015-2016 Contributors as noted in the AUTHORS file.
// This file is part of Thorium.
//
// Thorium is free software: you can redistribute it and/or modify
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

#pragma once

#include "../emission/system.h"
#include "../emission/malley_cpu.h"
#include "../emission/parallel_rays_cpu.h"

namespace cuda_emission
{
  bool is_supported();
}

namespace cuda_malley_emission
{
  using malley_cpu::params_t;

  emission::system_t* system_create();
}

namespace cuda_parallel_rays_emission
{
  using parallel_rays_emission_cpu::params_t;

  emission::system_t* system_create();
}