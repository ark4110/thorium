#include "tetrahedron.h"
#include "operations.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

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

  void tetrahedron_build_faces_indices(int *faces)
  {
    faces[0] = 0;
    faces[1] = 1;
    faces[2] = 2;

    faces[3] = 0;
    faces[4] = 2;
    faces[5] = 3;

    faces[6] = 0;
    faces[7] = 3;
    faces[8] = 1;

    faces[9] = 1;
    faces[10] = 3;
    faces[11] = 2;
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

  float tetrahedron_volume(const tetrahedron_t& object)
  {
    return tetrahedron_volume(object.points);
  }

  float tetrahedron_volume(const triangle_t *faces)
  {
    vec3 vertices[4];
    if (tetrahedron_build_from_faces(faces, vertices))
    {
      fprintf(stderr, "Degenerate tetrahedron found.\n");
      return NAN;
    }

    return tetrahedron_volume(vertices);
  }

  bool tetrahedron_has_adjacent_face(const tetrahedron_t& l, const tetrahedron_t& r)
  {
    float distance_scale = FLT_MAX;
    {
      face_t faces[8];
      tetrahedron_build_faces(l, faces);
      tetrahedron_build_faces(r, faces + 4);
      for (int fi = 0; fi != 8; ++fi)
        distance_scale = std::min<float>(distance_scale, triangle_least_side_square(faces[fi]));
    }

    vec3 lp[4];
    vec3 rp[4];

    memcpy(lp, l.points, sizeof(vec3) * 4);
    memcpy(rp, r.points, sizeof(vec3) * 4);

    float scale = 1.f / distance_scale;
    auto scale_less = [scale](vec3 l, vec3 r) {
      auto dist = l - r;
      if (dot(dist, dist) * scale < FLT_EPSILON)
        return false;
      return l < r;
    };

    std::sort(lp, lp + 4, scale_less);
    std::sort(rp, rp + 4, scale_less);
    vec3 out[4];
    const vec3* intersection = std::set_intersection(lp, lp + 4, rp, rp + 4, out, scale_less);
    return (intersection - out) >= 3;
  }

  bool is_tetra(const int* faces, int(&basis)[4])
  {
    int vertices[12];
    int* v= vertices;
    for (int fi = 0; fi != 12; ++fi, ++v, ++faces)
      *v = *faces;

    std::sort(vertices, vertices + 12);
    auto unique = std::unique(vertices, vertices + 12);
    if ((unique - vertices) != 4)
      return false;

    for (int i = 0; i != 4; ++i)
      basis[i] = vertices[i];

    return true;
  }

  void swap_normal(int* face)
  {
    int t = face[0];
    face[0] = face[1];
    face[1] = t;
  }

  bool tetrahedron_is_outside(const int* face, const int(&basis)[4], const math::vec3* vertices)
  {
    int bits = 1 + 2 + 4 + 8;
    for (int vi = 0; vi != 3; ++vi)
    {
      int vertex = face[vi];
      for (int i = 0; i != 4; ++i)
        bits ^= (vertex == basis[i]) ? (1 << i) : 0;
    }

    int found = ffs(bits) - 1;
    int other = (found + 1) % 4;
    auto direction = vertices[basis[found]] - vertices[basis[other]];

    math::vec3 base[3] = {vertices[face[0]], vertices[face[1]], vertices[face[2]]};
    auto normal = math::triangle_normal(base);

    return math::dot(normal, direction) < 0;
  }

  bool normalize_tetra(int* faces, const int(&basis)[4], const math::vec3* vertices)
  {
    bool swapped = false;
    for (int fi = 0; fi != 4; ++fi, faces += 3)
    {
      if (!tetrahedron_is_outside(faces, basis, vertices))
      {
        swapped = true;
        swap_normal(faces);
      }
    }
    return swapped;
  }

  tetrahedron_t make_simple_tetrahedron()
  {
    tetrahedron_t r;
    r.points[0] = {0, 0, 0};
    r.points[1] = {0.5f, 0.8660254, 0};
    r.points[2] = {1, 0, 0};
    r.points[3] = {0.5, 0.28867513f, 0.81649658f};
    return r;
  }
}
