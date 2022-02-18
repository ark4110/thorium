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


#include "conductive_cpu.h"
#include "../math/triangle.h"
#include "../math/operations.h"
#include <cstdlib>
#include <cstdio>
#include <assert.h>
#include <vector>

namespace conductive_equation
{
  class edges_cache_t : subject::face_graph_index_t
  {
  public:
    static int walk_method(const face_graph_index_t* system, subject::face_graph_walker walker, void* param)
    {
      return static_cast<const edges_cache_t*>(system)->walk(walker, param);
    }

    static void free_data_method(face_graph_index_t* system)
    {
      static_cast<edges_cache_t*>(system)->~edges_cache_t();
      free(system);
    }

    edges_cache_t(const int* face2mesh_index, face_graph_index_t* raw_index)
    {
      face2mesh = face2mesh_index;
      this->walk_func = &walk_method;
      this->free_data = &free_data_method;

      face_graph_walk_index(raw_index, &builder_func, this);
    }

    static int builder_func(int current_idx, int leaf_idx, int vertex_mapping, bool have_more, void* param)
    {
      return static_cast<edges_cache_t*>(param)->build(current_idx, leaf_idx, vertex_mapping);
    }

    int build(int l, int r, int vertex_mapping)
    {
      if (r == -1)
        return 0;

      if (face2mesh[l] == face2mesh[r])
        return 0;

      unsigned char lp[3];
      unsigned char rp[3];
      if (math::decode_vertex_mapping(vertex_mapping, lp, rp) < 2)
        return 0;
      records.push_back({ l, r, vertex_mapping });
      return 0;
    }

    int walk(subject::face_graph_walker walker, void* param) const
    {
      int r = 0;
      for (const auto& record : records)
        if ((r = walker(record.l, record.r, record.vertex_mapping, true, param)))
          return r;
      return 0;
    }

    const int* face2mesh;
    struct record_t
    {
      int l;
      int r;
      int vertex_mapping;
    };
    std::vector<record_t> records;
  };

  subject::face_graph_index_t* face_graph_mesh_edges_cache(const int* face2mesh, subject::face_graph_index_t* raw_index)
  {
    subject::face_graph_index_t* index = (subject::face_graph_index_t*)malloc(sizeof(edges_cache_t));
    new(index)edges_cache_t(face2mesh, raw_index);
    return index;
  }

  struct cpu_system_t : thermal_equation::system_t
  {
    params_t params;

    subject::scene_t* scene;
    subject::face_graph_index_t* graph;
    math::vec3* mesh_centers;
  };

  double calculate_alpha(const subject::scene_t* scene, double R, int l_mesh, int r_mesh)
  {
    auto c1 = mesh_C(scene, l_mesh);
    auto c2 = mesh_C(scene, r_mesh);
    return 1. / R * (c1 + c2) / c1 / c2;
  }

  /**
   * dF(t)/dt=F(t)
   * https://www.wolframalpha.com/input?i=dF%28t%29%2Fdt%3DF%28t%29
   *
   * F'(t) is a Dawson function
   * expansion at 0
   * https://www.wolframalpha.com/input?i=1%2F2+e%5E%28-t%5E2%29+sqrt%28%CF%80%29+erfi%28t%29
   */
  double dawson(double t)
  {
    auto c1 = 1.;
    auto c3 = -2. / 3;
    auto c5 = 4. / 15;
    auto c7 = -8. / 105;
    auto t1 = t;
    auto t3 = pow(t, 3);
    auto t5 = pow(t, 5);
    auto t7 = pow(t, 7);
    // Remove o(t7)?
    return c1 * t1 + c3 * t3 + c5 * t5 + c7 * t7;
  }

  int init(cpu_system_t* system, params_t* params)
  {
    system->params = *params;
    
    system->scene = 0;
    system->graph = 0;
    system->mesh_centers = 0;
    
    return THERMAL_EQUATION_OK;
  }

  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    subject::face_graph_index_free(system->graph);

    free(system->mesh_centers);
    system->mesh_centers = 0;

    return THERMAL_EQUATION_OK;
  }

  math::vec3 mesh_center(const subject::scene_t* scene, int mesh_idx)
  {
    const subject::mesh_t& mesh = scene->meshes[mesh_idx];
    math::vec3 center = triangle_center(scene->faces[mesh.first_idx]);
    for (int f = 1; f != mesh.n_faces; ++f)
      center += triangle_center(scene->faces[mesh.first_idx + f]);
    center /= (float)mesh.n_faces;
    return center;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;

    subject::face_graph_index_t* raw_graph = subject::face_graph_index_create(scene->faces, scene->n_faces);
    system->graph = face_graph_mesh_edges_cache(scene->face_to_mesh, raw_graph);
    face_graph_index_free(raw_graph);

    system->mesh_centers = (math::vec3*) calloc(scene->n_meshes, sizeof(math::vec3));
    for (int m = 0; m != scene->n_meshes; ++m)
      system->mesh_centers[m] = mesh_center(scene, m);

    return THERMAL_EQUATION_OK;
  }

  struct graph_walker_param_t
  {
    cpu_system_t* system;
    thermal_equation::task_t* task;
  };

  int graph_walker(int l, int r, int vertex_mapping, bool have_more, graph_walker_param_t* param)
  {
    if (r == -1)
      return 0;

    int l_mesh_idx = param->system->scene->face_to_mesh[l];
    int r_mesh_idx = param->system->scene->face_to_mesh[r];

    if (l_mesh_idx == r_mesh_idx)
      return 0;

    unsigned char lp[3];
    unsigned char rp[3];
    int n_points = math::decode_vertex_mapping(vertex_mapping, lp, rp);
    if (n_points < 2)
      return 0;

    const bool edge_contact = n_points == 2;

    subject::scene_t* scene = param->system->scene;

    const subject::material_t& l_material = mesh_material(scene, l_mesh_idx);
    const subject::material_t& r_material = mesh_material(scene, r_mesh_idx);

    /// @todo Drop such graph edges.
    if (edge_contact && (l_material.shell.thickness == 0 || r_material.shell.thickness))
      return 0;

    /// @todo Validate during graph build.
    if (!edge_contact && (l_material.shell.thickness > 0 || r_material.shell.thickness > 0))
    {
      fprintf(stderr, "Area contact of not solid objects.\n");
      assert(false);
      return -1;
    }

    const math::vec3* mesh_centers = param->system->mesh_centers;

    const subject::face_t& l_face = scene->faces[l];
    math::vec3 median = edge_contact ? ((l_face.points[lp[0]] + l_face.points[lp[1]]) / 2.f)
            : math::triangle_center(l_face);

    const math::vec3& l_center = mesh_centers[l_mesh_idx];
    double l_dist = norm(median - l_center);

    const math::vec3& r_center = mesh_centers[r_mesh_idx];
    double r_dist = norm(median - r_center);

    float area = 0;
    if (edge_contact)
    {
      float thickness = std::fminf(l_material.shell.thickness, r_material.shell.thickness);
      float edge_length = norm(l_face.points[lp[0]] - l_face.points[lp[1]]);
      area = thickness * edge_length;
    }
    else
    {
      area = math::triangle_area(l_face);
    }

    double conductivity_resistance = (l_dist / l_material.shell.thermal_conductivity + r_dist / r_material.shell.thermal_conductivity) / area;

    float l_temp = param->task->temperatures[l_mesh_idx];
    float r_temp = param->task->temperatures[r_mesh_idx];

    double flow = (r_temp - l_temp) / conductivity_resistance;
    assert(!std::isnan(flow) && !std::isinf(flow) && "bad value detected");

    const int steps = param->system->params.integrations_steps;
    if (steps > 1)
    {
      auto step = param->system->params.step;
      double dt = step / steps;
      auto l_rC = 1. / mesh_C(scene, l_mesh_idx);
      auto r_rC = 1. / mesh_C(scene, r_mesh_idx);
      double integrated_flow = 0;
      for (int i = 0; i != steps; ++i)
      {
        auto dQ = flow * dt;
        integrated_flow += dQ;
        l_temp += dQ * l_rC;
        r_temp -= dQ * r_rC;
        flow = (r_temp - l_temp) / conductivity_resistance;
      }
      flow = integrated_flow / step;
    }

    if (flow > 0)
    {
      param->task->absorption[l_mesh_idx] += flow;
      param->task->emission[r_mesh_idx] += flow;
    }
    else if (flow < 0)
    {
      param->task->emission[l_mesh_idx] += -flow;
      param->task->absorption[r_mesh_idx] += -flow;
    }

    return 0;
  }
  
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    graph_walker_param_t param = { system, task };
    int r = subject::face_graph_walk_index(system->graph, (subject::face_graph_walker)graph_walker, &param);
    return r == 0 ? THERMAL_EQUATION_OK : THERMAL_EQUATION_ERROR;
  }

  int update(cpu_system_t* system)
  {
    return THERMAL_EQUATION_OK;
  }

  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system))&update,
    (int(*)(thermal_equation::system_t* system, thermal_equation::task_t* task))&calculate,
  };

  thermal_equation::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }

}
