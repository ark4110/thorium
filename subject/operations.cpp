// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
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

#include "system.h"
#include "../math/triangle.h"
#include <cstdlib>

namespace subject
{
  void scene_mesh_compose(scene_t* scene, int n_first_idx, int n_faces)
  {
    scene->meshes = (mesh_t*)realloc(scene->meshes, (scene->n_meshes + 1) * sizeof(mesh_t));
    mesh_t& new_mesh = scene->meshes[scene->n_meshes];
    new_mesh.first_idx = n_first_idx;
    new_mesh.n_faces = n_faces;
    new_mesh.material_idx = -1;
    new_mesh.name = 0;
    ++scene->n_meshes;
  }

  void scene_mesh_translate(scene_t* scene, int n_mesh, math::vec3 translation)
  {
    mesh_t& mesh = scene->meshes[n_mesh];
    for (int i = mesh.first_idx; i != mesh.first_idx + mesh.n_faces; ++i)
      scene->faces[i] = math::triangle_translate(scene->faces[i], translation);
  }
  void scene_mesh_rotate(scene_t* scene, int n_mesh, math::vec3 from, math::vec3 to)
  {
    mesh_t& mesh = scene->meshes[n_mesh];
    for (int i = mesh.first_idx; i != mesh.first_idx + mesh.n_faces; ++i)
      scene->faces[i] = math::triangle_rotate(scene->faces[i], from, to);
  }

  bool scenes_mesh_faces_near_enough(const scene_t* a, const scene_t* b, int ia, int ib)
  {
    const mesh_t& ma = a->meshes[ia];
    const mesh_t& mb = b->meshes[ib];

    for (int i = ma.first_idx; i != ma.first_idx + ma.n_faces; ++i)
    {
      bool found_near_enough = false;
      for (int j = mb.first_idx; !found_near_enough && j != mb.first_idx + mb.n_faces; ++j)
        found_near_enough = math::near_enough(a->faces[i], b->faces[j]);
      if (!found_near_enough)
        return false;
    }

    return true;
  }
}
