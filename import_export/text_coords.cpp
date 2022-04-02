#include "text_coords.h"
#include <cstdio>
#include <cfloat>

namespace texture_coordinates
{
  void find_min_max(const subject::scene_t* scene, const float* temperatures, float& min, float& max)
  {
    min = FLT_MAX;
    max = -FLT_MAX;
    for (int mi = 0; mi != scene->n_meshes; ++mi)
    {
      auto t = temperatures[mi];
      min = min < t ? min : t;
      max = max > t ? max : t;
    }
  }

  void generate(const subject::scene_t* scene, const float* temperatures, float* coords)
  {
    float min;
    float max;
    find_min_max(scene, temperatures, min, max);
    const float scale = 1.f / (max - min);

    for (int vi = 0; vi != scene->n_vertices; ++vi)
    {
      auto offset = scene->vertex_to_mesh_offsets[vi];
      auto n_meshes = scene->vertex_to_mesh_offsets[vi + 1] - offset;
      float total_C = 0;
      float total = 0;
      for (int mi = 0; mi != n_meshes; ++mi)
      {
        auto mesh_idx = scene->vertex_to_mesh_index[offset + mi];
        auto C = subject::mesh_C(scene, mesh_idx);
        total_C += C;
        total += C * temperatures[mesh_idx];
      }
      auto avg = total / total_C;
      float coord = (avg - min) * scale;
      coords[vi] = coord;
    }
  }

  int dump(const char* filename, int n_vertices, const float* coordinates)
  {
    FILE* f = fopen(filename, "wb");
    if (!f)
    {
      fprintf(stderr, "Failed to open %s.\n", filename);
      return -1;
    }

    for (int vi = 0; vi != n_vertices; ++vi)
    {
      fprintf(f, "vt %f 0\n", coordinates[vi]);
    }

    fclose(f);
  }
}
