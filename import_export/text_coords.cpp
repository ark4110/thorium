#include "text_coords.h"
#include <cstdio>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <map>
#include "ext/tinycolormap/include/tinycolormap.hpp"

namespace texture_coordinates
{
  tinycolormap::ColormapType parse_colormap_name(const char* name_)
  {
    using tinycolormap::ColormapType;
    static const std::map<std::string, tinycolormap::ColormapType> types =
            {
                    {"parula", ColormapType::Parula},
                    {"heat", ColormapType::Heat},
                    {"jet", ColormapType::Jet},
                    {"turbo", ColormapType::Turbo},
                    {"hot", ColormapType::Hot},
                    {"gray", ColormapType::Gray},
                    {"magma", ColormapType::Magma},
                    {"inferno", ColormapType::Inferno},
                    {"plasma", ColormapType::Plasma},
                    {"viridis", ColormapType::Viridis},
                    {"cividis", ColormapType::Cividis},
                    {"github", ColormapType::Github},
                    {"cubehelix", ColormapType::Cubehelix}
            };

    std::string name(name_);
    for (char& c: name)
      c = std::tolower(c);

    auto found = types.find(name);
    if (found == types.end())
    {
      fprintf(stderr, "Can't find colormap %s. Using default Heat colormap.\n", name_);
      return ColormapType::Heat;
    }

    return found->second;
  }

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
    fprintf(stderr, "Scene min/max temperatures %f/%f.\n", min, max);
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
    return 0;
  }

  struct rgb_t
  {
    float r;
    float g;
    float b;
  };

  rgb_t ratio_to_rgb(float ratio)
  {
    auto r = tinycolormap::GetColor(ratio, tinycolormap::ColormapType::Hot);
    return {(float)r.r(), (float)r.g(), (float)r.b()};
  }

  void dump_rgb_vertices(FILE* out, const subject::scene_t* scene, const float* ratios, tinycolormap::ColormapType colormap)
  {
    for (int vi = 0; vi != scene->n_vertices; ++vi)
    {
      auto coords = scene->vertices[vi];
      auto rgb = tinycolormap::GetColor(ratios[vi], colormap);
      fprintf(out, "v %f %f %f %f %f %f\n", coords.x, coords.y, coords.z, rgb.r(), rgb.g(), rgb.b());
    }
  }

  void dump_meshes(FILE* out, const subject::scene_t* scene, params_t params)
  {
    for (int mi = 0; mi != scene->n_meshes; ++mi)
    {
      const auto& mesh = scene->meshes[mi];
      fprintf(out, "g %s\n", mesh.name);
      for (int fi = 0; fi != mesh.n_faces; ++fi)
      {
        int face_idx = mesh.first_idx + fi;
        if (!params.only_emission || !scene->blind_faces[face_idx])
        {
          const int *face_vertices = scene->faces_indices + 3 * face_idx;
          fprintf(out, "f %d %d %d\n", 1 + face_vertices[0], 1 + face_vertices[1], 1 + face_vertices[2]);
        }
      }
      fprintf(out, "\n");
    }
  }

  int dump_scene_with_colors(const char* filename, const char* colormap_name, const subject::scene_t* scene, const float* temperatures, params_t params)
  {
    FILE* f = fopen(filename, "wb");
    if (!f)
    {
      fprintf(stderr, "Failed to open %s.\n", filename);
      return -1;
    }

    auto colormap = parse_colormap_name(colormap_name);

    std::vector<float> ratio(scene->n_vertices);
    generate(scene, temperatures, ratio.data());
    dump_rgb_vertices(f, scene, ratio.data(), colormap);
    fprintf(f, "\n");

    dump_meshes(f, scene, params);

    fclose(f);
    return 0;
  }
}
