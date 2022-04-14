#include "normalization.h"
#include <algorithm>
#include <strings.h>
#include <cstdio>
#include "../math/triangle.h"
#include "../math/operations.h"
#include "../math/tetrahedron.h"

namespace obj_import
{
  void normalize(int n_faces, int* faces, const math::vec3* vertices)
  {
    if (n_faces != 3)
      return;

    int tetra_basis[4];
    if (!math::is_tetra(faces, tetra_basis))
      return;

    int count = 0;
    if (normalize_tetra(faces, tetra_basis, vertices))
      ++count;
    if (count)
      fprintf(stderr, "Normalized %d tetrahedrons normals.\n", count);
  }

  void normalize(int n_meshes, const subject::mesh_t* meshes, const math::vec3* vertices, int* faces)
  {
    for (int mi = 0; mi != n_meshes; ++mi)
    {
      const auto& mesh = meshes[mi];
      normalize(mesh.n_faces, faces + mesh.first_idx, vertices);
    }
  }
}
