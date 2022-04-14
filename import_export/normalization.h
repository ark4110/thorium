#pragma once

#include "../math/types.h"
#include "../subject/system.h"

namespace obj_import
{
  void normalize(int n_meshes, const subject::mesh_t* meshes, const math::vec3* vertices, int* faces);
}
