#include "normalization.h"
#include <algorithm>
#include <cstdio>
#include "../math/triangle.h"
#include "../math/operations.h"
#include "../math/tetrahedron.h"

namespace obj_import
{
  bool normalize(int n_faces, int* faces, const math::vec3* vertices)
  {
    if (n_faces != 4)
      return false;

    int tetra_basis[4];
    if (!math::is_tetra(faces, tetra_basis))
      return false;

    return normalize_tetra(faces, tetra_basis, vertices);
  }

  bool normalize(const subject::mesh_t& mesh, const math::vec3* vertices, int* faces)
  {
    return normalize(mesh.n_faces, faces + 3ll * mesh.first_idx, vertices);
  }

  void normalize(int n_meshes, const subject::mesh_t* meshes, const math::vec3* vertices, int* faces)
  {
    int count = 0;
    for (int mi = 0; mi != n_meshes; ++mi)
    {
      const auto& mesh = meshes[mi];
      count += normalize(mesh, vertices, faces) ? 1 : 0;
    }

    if (count)
      fprintf(stderr, "Normalized %d tetrahedrons normals of %d meshes.\n", count, n_meshes);
  }
}
