#include "tetrahedron.h"
#include "operations.h"

#include <cstdio>

namespace math
{
  int tetrahedron_build_from_faces(const triangle_t *faces, vec3 *vertices)
  {
    vertices[0] = faces[0].points[0];
    vertices[1] = faces[0].points[1];
    vertices[2] = faces[0].points[2];

    int n_found = 0;
    for (int fi = 1; fi != 4; ++fi)
    {
      for (int pi = 0; pi != 3; ++pi)
      {
        const vec3 &point = faces[fi].points[pi];
        if (triangle_find_adjacent_vertex(faces[0], point) == -1)
        {
          vertices[3] = point;
          ++n_found;
        }
      }
    }

    return n_found == 3 ? 0 : -1;
  }

  void tetrahedron_build_faces(const tetrahedron_t &object, triangle_t *faces)
  {
    const vec3 *vertices = object.points;

    faces[0].points[0] = vertices[0];
    faces[0].points[1] = vertices[1];
    faces[0].points[2] = vertices[2];

    faces[1].points[0] = vertices[0];
    faces[1].points[1] = vertices[2];
    faces[1].points[2] = vertices[3];

    faces[2].points[0] = vertices[0];
    faces[2].points[1] = vertices[3];
    faces[2].points[2] = vertices[1];

    faces[3].points[0] = vertices[1];
    faces[3].points[1] = vertices[3];
    faces[3].points[2] = vertices[2];
  }

  float tetrahedron_volume(const vec3 *vertices)
  {
    vec3 a = vertices[1] - vertices[0];
    vec3 b = vertices[2] - vertices[0];
    vec3 c = vertices[3] - vertices[0];

    return 0.16666666666f * fabs(dot(cross(a, b), c));
  }

  float tetrahedron_volume(const triangle_t *faces)
  {
    vec3 vertices[4];
    if (int r = tetrahedron_build_from_faces(faces, vertices))
    {
      fprintf(stderr, "Degenerate tetrahedron found.\n");
      return NAN;
    }

    return tetrahedron_volume(vertices);
  }
}
