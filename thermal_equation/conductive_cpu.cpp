// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
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


#include "conductive_cpu.h"
#include "../math/triangle.h"
#include "../math/operations.h"
#include <cstdlib>
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
    int* face_to_mesh_index;
    math::vec3* mesh_centers;
  };

  int init(cpu_system_t* system, params_t* params)
  {
    system->params = *params;
    
    system->scene = 0;
    system->graph = 0;
    system->face_to_mesh_index = 0;
    system->mesh_centers = 0;
    
    return THERMAL_EQUATION_OK;
  }

  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    subject::face_graph_index_free(system->graph);

    free(system->face_to_mesh_index);
    system->face_to_mesh_index = 0;

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

    build_face_to_mesh_index(scene->n_faces, scene->n_meshes, scene->meshes, &system->face_to_mesh_index);

    subject::face_graph_index_t* raw_graph = subject::face_graph_index_create(scene->faces, scene->n_faces);
    system->graph = face_graph_mesh_edges_cache(system->face_to_mesh_index, raw_graph);
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

    int l_mesh_idx = param->system->face_to_mesh_index[l];
    int r_mesh_idx = param->system->face_to_mesh_index[r];

    if (l_mesh_idx == r_mesh_idx)
      return 0;

    unsigned char lp[3];
    unsigned char rp[3];
    if (math::decode_vertex_mapping(vertex_mapping, lp, rp) < 2)
      return 0;

    subject::scene_t* scene = param->system->scene;
    const math::vec3* mesh_centers = param->system->mesh_centers;

    const subject::face_t& l_face = scene->faces[l];
    float edge_length = norm(l_face.points[lp[0]] - l_face.points[lp[1]]);
    math::vec3 median = (l_face.points[lp[0]] + l_face.points[lp[1]]) / 2.f;
    const math::vec3& l_center = mesh_centers[l_mesh_idx];
    float l_dist = norm(median - l_center);

    const math::vec3& r_center = mesh_centers[r_mesh_idx];
    float r_dist = norm(median - r_center);

    const subject::material_t& l_matertial = mesh_material(scene, l_mesh_idx);
    const subject::material_t& r_matertial = mesh_material(scene, r_mesh_idx);
    
    float thickness = std::fminf(l_matertial.shell.thickness, r_matertial.shell.thickness);
    float area = thickness * edge_length;
    float conductivity_resistance = (l_dist / l_matertial.shell.thermal_conductivity + r_dist / r_matertial.shell.thermal_conductivity) / area;

    float l_temp = param->task->temperatures[l_mesh_idx];
    float r_temp = param->task->temperatures[r_mesh_idx];

    float flow = (r_temp - l_temp) / conductivity_resistance;
    if (flow > 0)
    {
      param->task->absorption[l_mesh_idx] += flow;
      param->task->emission[r_mesh_idx] += flow;
    }
    else
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

  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system, thermal_equation::task_t* task))&calculate,
  };

  thermal_equation::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }

}
