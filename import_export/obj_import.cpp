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

/**
 * This module contains functionality of loading obj-files (Wavefront).
 */

#include "obj_import.h"
#include "../math/operations.h"
#include "../math/mat.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <set>
#include <tuple>

#ifdef _WIN32
#pragma warning(disable:4996)
#endif

namespace obj_import
{
  /// @brief Local structure representing vertex indices for given face (polygon).
  struct idx_face_t
  {
    int indices[3];
  };

  int parse_material(std::istream& stream, std::string header, unsigned& line, std::vector<subject::material_t>& materials, std::map<std::string, int>& library)
  {
    std::map<std::string, bool> known_unknown_attributes = {
            {"Ns", true},
            {"Ka", true},
            {"Kd", true},
            {"Ks", true},
            {"Ke", true},
            {"Ni", true},
            {"d", true},
            {"illum", true}
    };

    if (strncmp("newmtl", header.c_str(), 6) != 0)
    {
      fprintf(stderr, "Failed to parse newmtl at line %d\n", line);
      return -OBJ_IMPORT_FORMAT_ERROR;
    }

    std::string name = header.substr(7);

    subject::material_t m = subject::material_t();

    if (name.length() >= sizeof(m.name))
    {
      fprintf(stderr, "Material name '%s' is too long at line %d\n", name.c_str(), line);
      return -OBJ_IMPORT_FORMAT_ERROR;
    }

    strncpy(m.name, name.c_str(), ::min(sizeof(m.name), name.length()));

    typedef std::map<std::string, float*> MaterialPropertiesMap;
    MaterialPropertiesMap materialProperties;
    materialProperties["shell.density"] = &m.shell.density;
    materialProperties["shell.heat_capacity"] = &m.shell.heat_capacity;
    materialProperties["shell.thermal_conductivity"] = &m.shell.thermal_conductivity;
    materialProperties["shell.thickness"] = &m.shell.thickness;
    materialProperties["front.specular_reflectance"] = &m.front.optical.specular_reflectance;
    materialProperties["front.diffuse_reflectance"] = &m.front.optical.diffuse_reflectance;
    materialProperties["front.absorbance"] = &m.front.optical.absorbance;
    materialProperties["front.transmittance"] = &m.front.optical.transmittance;
    materialProperties["front.emissivity"] = &m.front.emission.emissivity;
    materialProperties["rear.specular_reflectance"] = &m.rear.optical.specular_reflectance;
    materialProperties["rear.diffuse_reflectance"] = &m.rear.optical.diffuse_reflectance;
    materialProperties["rear.absorbance"] = &m.rear.optical.absorbance;
    materialProperties["rear.transmittance"] = &m.rear.optical.transmittance;
    materialProperties["rear.emissivity"] = &m.rear.emission.emissivity;

    typedef std::set<std::string> StringSet;
    StringSet parsedParameters;

    std::string str;
    while (getline(stream, str) && parsedParameters.size() != materialProperties.size())
    {
      ++line;
      if (str == "")
        break;

      if (str[0] == '#')
        continue;

      std::string::size_type pos = str.find_first_of(" \t");
      if (pos + 1 >= str.length())
      {
        fprintf(stderr, "Invalid material definition format at line %d\n", line);
        return -OBJ_IMPORT_FORMAT_ERROR;
      }

      std::string propertyName = str.substr(0, pos);
      std::string propertyValue = str.substr(pos + 1);

      MaterialPropertiesMap::const_iterator found = materialProperties.find(propertyName);
      if (found == materialProperties.end())
      {
        bool& known = known_unknown_attributes[propertyName];
        if (!known)
        {
          known = true;
          fprintf(stderr, "Skip unknown material property '%s'. First at line %d\n", propertyName.c_str(), line);
        }
        continue;
      }

      if (sscanf(propertyValue.c_str(), "%f", found->second) != 1)
      {
        fprintf(stderr, "Failed to parse material '%s' value '%s' at line %d\n", propertyName.c_str(), propertyValue.c_str(), line);
        return -OBJ_IMPORT_MATERIAL_INVALID_PARAMETER_VALUE;
      }

      parsedParameters.insert(propertyName);
    }

    if (parsedParameters.size() != materialProperties.size())
    {
      std::string missingProperties;
      for (MaterialPropertiesMap::const_iterator checkName = materialProperties.begin(); checkName != materialProperties.end(); ++checkName)
      {
        if (parsedParameters.find(checkName->first) == parsedParameters.end())
        {
          missingProperties += missingProperties.empty() ? "" : ", ";
          missingProperties += "'" + checkName->first + "'";
        }
      }

      fprintf(stderr, "Not enough (%zu) material parameters for material '%s' at line %d, missing properties are %s\n", materialProperties.size() - parsedParameters.size(), name.c_str(), line, missingProperties.c_str());
      return -OBJ_IMPORT_MATERIAL_NOT_ENOUGH_PARAMETERS;
    }

    materials.push_back(m);
    library[name] = (int)(materials.size() - 1);

    return 0;
  }

  int material_lib(const char* filename, std::vector<subject::material_t>& materials, std::map<std::string, int>& library)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      fprintf(stderr, "Material library file '%s' not found.\n", filename);
      return -OBJ_IMPORT_FILE_ERROR;
    }

    std::string str;
    unsigned line = 0;

    while (getline(file, str))
    {
      ++line;
      if (str == "" || str[0] == '#')
      {
        continue;
      }

      switch (str[0])
      {
        case 'n':
        {
          if (int r = parse_material(file, str, line, materials, library))
            return r;
        }
          break;
      }
    }

    return OBJ_IMPORT_OK;
  }

  struct frame_t
  {
    std::map<std::string, math::mat44> transformations;
  };

  std::tuple<int, std::map<int, frame_t>, std::set<std::string>, std::set<std::string>, std::map<std::string, std::string>>
   animation(const char* filename, subject::scene_t* scene)
  {
    int r = OBJ_IMPORT_OK;

    std::map<int, frame_t> frames;
    std::set<std::string> roots;
    std::map<std::string, std::string> hierarchy;
    std::set<std::string> dummies;
    
    int current_frame = -1;
    float frame_time_scale = 1.f;

    std::ifstream file(filename);
    if (!file.is_open())
      fprintf(stderr, "No scene animation '%s' found.\n", filename);
    else
    {
      typedef std::function<bool(const char*)> action_t;

      std::map<std::string, action_t> actions;

      actions["object_root"] = [&](const char* line) {
        std::string obj, parent;
        std::istringstream(line) >> obj >> parent;
        roots.insert(parent);
        hierarchy[obj] = parent;
        return obj != "" && parent != "";
      };

      actions["dummy_object"] = [&](const char* line) {
        dummies.insert(line);
        return true;
      };

      actions["frame"] = [&](const char* line) {
        return sscanf(line, "%d", &current_frame) == 1;
      };

      actions["frame_time_scale"] = [&](const char* line) {
        return sscanf(line, "%f", &frame_time_scale) == 1;
      }; 

      actions["transform"] = [&](const char* line) {
        const char* delim = strchr(line, ' ');
        if (!delim)
          return false;
        std::string name(line, delim - line);
        return load_from_string(delim + 1, frames[current_frame].transformations[name]);
      };

      std::string line;
      while (getline(file, line))
      {
        if (line == "" || line[0] == '#')
          continue;

        const char* str = line.c_str();
        const char* delim = strchr(str, ' ');
        std::string token = delim ? std::string(str, delim - str) : line;
        const char* params = delim ? delim + 1 : "";

        auto found = actions.find(token);
        if (found == actions.end())
        {
          fprintf(stderr, "ERROR: Unknown animation action in line %s.\n", line.c_str());
          continue;
        }

        const action_t& action = found->second;
        if (!action(params))
        {
          fprintf(stderr, "ERROR: Failed to parse animation at line %s.\n", line.c_str());
          r = -OBJ_IMPORT_FILE_ERROR;
          break;
        }
      }
    }

    int n_groups = (int)roots.size();
    int n_frames = frames.empty() ? 0 : frames.rbegin()->first + 1;
    if (!frames.empty())
    {
      for (int frame = frames.rbegin()->first; frame >= 0; --frame)
      {
        if (frames.find(frame) == frames.end())
        {
          fprintf(stderr, "WARNING: Extrapolating animation frame %d\n", frame);
          frames[frame] = frames[frame + 1];
        }
      }
    }

    animation_free(scene->animation);

		if (n_frames)
		{
			scene->animation = subject::animation_create(n_groups, n_frames);
			scene->animation->frame_time_step = frame_time_scale;
		}

    for (const auto& frame_iter : frames)
    {
      int n_frame = frame_iter.first;
      const frame_t& frame = frame_iter.second;
      
      int n_group = 0;
      for (const auto& root : roots)
      {
        auto found = frame.transformations.find(root);
        if (found == frame.transformations.end())
        {
          fprintf(stderr, "ERROR: No transformation for object %s at frame %d\n", root.c_str(), n_frame);
          r = -OBJ_IMPORT_FORMAT_ERROR;
          break;
        }
        const math::mat44& transformation = found->second;
        scene->animation->translations[n_frame * n_groups + n_group] = translation(transformation);
        scene->animation->rotations[n_frame * n_groups + n_group] = rotation(transformation);
        ++n_group;
      }

      if (r)
        break;
    }

    if (n_frames && r == OBJ_IMPORT_OK)
    {
      fprintf(stderr, "INFO: Loaded %d animation frames\n", n_frames);
      fprintf(stderr, "INFO: Loaded %d animation groups\n", n_groups);
    }

    return std::make_tuple(r, frames, roots, dummies, hierarchy);
  }

  int scene(const char* filename, subject::scene_t** result)
  {
    *result = subject::scene_create();
    return scene(filename, *result);
  }

  std::string get_animation_filename(std::string basename)
  {
    return basename.substr(0, basename.find_last_of('.')) + ".anim";
  }

  int scene(const char* filename, subject::scene_t* scene)
  {
    auto animation_result = animation(get_animation_filename(filename).c_str(), scene);
    if (std::get<0>(animation_result))
      return std::get<0>(animation_result);

    std::map<int, frame_t> frames = std::move(std::get<1>(animation_result));
    std::set<std::string> roots = std::move(std::get<2>(animation_result));
    std::set <std::string> dummies = std::move(std::get<3>(animation_result));
    std::map<std::string, std::string> parents = std::move(std::get<4>(animation_result));
    std::multimap<std::string, std::string> children;
    for (const auto& i : parents)
      children.insert(std::make_pair(i.second, i.first));

    std::ifstream file(filename);
    if (!file.is_open())
    {
      fprintf(stderr, "Scene file '%s' not found.\n", filename);
      return -OBJ_IMPORT_FILE_ERROR;
    }

    std::vector<math::vec3> vertices;
    std::vector<idx_face_t> loaded_faces;
    std::string current_mesh_name;
    std::map<std::string, subject::mesh_t> meshes;
    std::vector<subject::material_t> materials;
    typedef std::map<std::string, int> MaterialMap;
    MaterialMap material2IndexMap;
    
    std::string str;
    unsigned line = 0;

    while (getline(file, str))
    {  
      ++line;
      if (str == "" || str[0] == '#')
      {
        continue;
      }

      if (str.length() < 3)
        continue;

      switch (str[0])
      {
      case 'l': // Some unknown obj directive
        continue;

      case 'v':
      {        
        if (str[1] == 'n' || str[1] == 't')
          continue;
        math::vec3 vertex;
        int count = sscanf(str.c_str() + 1, "%f %f %f", &(vertex.x), &(vertex.y), &(vertex.z));
        if (count == 3)
        {
          vertices.push_back(vertex);
        }
        else
        {
          return -OBJ_IMPORT_FORMAT_ERROR;
        }
      }
      break;

      case 'f':
      {
        int d;
        int i0, i1, i2;
        i0 = i1 = i2 = 0;
        int count = sscanf(str.c_str() + 1, "%d %d %d", &i0, &i1, &i2);
        if (count != 3)
        {
          count = sscanf(str.c_str() + 1, "%d//%d %d//%d %d//%d", &i0, &d, &i1, &d, &i2, &d);
          if (count != 6)
          {
            count = sscanf(str.c_str() + 1, "%d/%d/%d %d/%d/%d %d/%d/%d", &i0, &d, &d, &i1, &d, &d, &i2, &d, &d);
            if (count != 9)
            {
              fprintf(stderr, "Bad face description '%s' at line %d\n", str.c_str(), line);
              return -OBJ_IMPORT_FORMAT_ERROR;
            }
          }
        }

        idx_face_t face = { i0 - 1, i1 - 1, i2 - 1 };
        loaded_faces.push_back(face);
      }
      break;

      // Ignore shading smoothing group
      case 's':
        continue;

      case 'o':
      case 'g':
      { 
        if (current_mesh_name != "")
          meshes[current_mesh_name].n_faces = (int)(loaded_faces.size() - meshes[current_mesh_name].first_idx);
        
        current_mesh_name = str.substr(2);
        if (current_mesh_name == "")
        {
          char buf[100];
          sprintf(buf, "mesh_%d", (int)meshes.size());
          current_mesh_name = buf;

          fprintf(stderr, "WARNING: Autonaming mesh %s\n", current_mesh_name.c_str());
        }

        subject::mesh_t& mesh = meshes[current_mesh_name];

        mesh.name = strdup(current_mesh_name.c_str());
        mesh.first_idx = (int)loaded_faces.size();
        mesh.material_idx = 0;
      }
      break;

      case 'u':
      {
        std::string::size_type pos = str.find_first_of(" \t");
        if (pos == std::string::npos)
        {
          fprintf(stderr, "Failed to parse line %d '%s'\n", line, str.c_str());
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        std::string token = str.substr(0, pos);
        if (token == "usemtl")
        {
          std::string name = str.substr(pos + 1);
          MaterialMap::const_iterator found = material2IndexMap.find(name);
          if (found == material2IndexMap.end())
          {
            fprintf(stderr, "Failed to find material reference '%s' at line %d\n", name.c_str(), line);
            return -OBJ_IMPORT_MATERIAL_NOT_DEFINED;
          }

          meshes[current_mesh_name].material_idx = found->second;
        }        
        else
        {
          fprintf(stderr, "Invalid material token at line %d '%s'\n", line, str.c_str());
          return -OBJ_IMPORT_FORMAT_ERROR;
        }
      }
      break;

      case 'm':
      {
        if (strncmp("mtllib", str.c_str(), 6) != 0)
        {
          fprintf(stderr, "Failed to parse mtllib at line %d\n", line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        std::string library_name = str.substr(7);
        std::string library_path = library_name;
        
        if (library_name.find("/\\") > library_name.length())
        {
          std::string base_name(filename);
          size_t dir_pos = base_name.find_last_of("/\\");
          if (dir_pos < base_name.length())
            base_name = base_name.substr(0, dir_pos + 1);
          else
            base_name = "";
          std::string library_name = str.substr(7);
          library_path = base_name + library_name;
        }

        fprintf(stderr, "Loading material lib '%s'\n", library_path.c_str());

        if (int r = material_lib(library_path.c_str(), materials, material2IndexMap))
        {
          return r;
        }
      }
      break;

      case 'n':
      {
        if (strncmp("newmtl", str.c_str(), 6) != 0)
        {
          fprintf(stderr, "Failed to parse newmtl at line %d\n", line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        if (int r = parse_material(file, str, line, materials, material2IndexMap))
          return r;
      }
      break;

      default:
        fprintf(stderr, "Unknown directive '%s' at line %d\n", str.c_str(), line);
        continue;
        break;
      }    
    }

    meshes[current_mesh_name].n_faces = (int)loaded_faces.size() - meshes[current_mesh_name].first_idx;

    file.close();
    
    // Triangulation to vertices from face indices.
    scene->n_faces = (int)loaded_faces.size();
    scene->faces = (subject::face_t *)malloc(scene->n_faces * sizeof(subject::face_t));
    int n_current_face = 0;

    scene->n_meshes = ::max(1, (int)meshes.size());
    scene->meshes = (subject::mesh_t *)malloc(sizeof(subject::mesh_t) * scene->n_meshes);
    int n_current_mesh = 0;

    std::vector<std::vector<int>> vertex_to_mesh(vertices.size());

    auto add_mesh = [&n_current_face, &n_current_mesh, &loaded_faces, &scene, &vertices, &vertex_to_mesh](subject::mesh_t& mesh)
    {
      int mesh_first_idx = n_current_face;
      for (int f = mesh.first_idx; f != mesh.first_idx + mesh.n_faces; ++f)
      {
        idx_face_t idx_face = loaded_faces[f];
        scene->faces[n_current_face++] = {
                vertices[idx_face.indices[0]],
                vertices[idx_face.indices[1]],
                vertices[idx_face.indices[2]]
        };

        for (auto vertex_idx: idx_face.indices)
          vertex_to_mesh[vertex_idx].push_back(n_current_mesh);
      }
      mesh.first_idx = mesh_first_idx;
      scene->meshes[n_current_mesh++] = mesh;
    };

    // Write not animated geometry
    for (auto& mesh_iter : meshes)
    {
      const std::string& mesh_name = mesh_iter.first;
      if (parents.find(mesh_name) != parents.end())
        continue;

      subject::mesh_t& mesh = mesh_iter.second;
      add_mesh(mesh);
    }

    if (scene->animation)
      scene->animation->n_static_faces = n_current_face;

    int current_group = 0;
    // Write animated geometry
    for (const auto& root_name : roots)
    {
      auto range = children.equal_range(root_name);
      int n_group_start_face = n_current_face;
      scene->animation->groups[current_group].n_offset = n_group_start_face;
      for (; range.first != range.second; ++range.first)
      {
        const std::string& child_name = range.first->second;
        auto found = meshes.find(child_name);
        if (found == meshes.end())
        {
          if (dummies.find(child_name) != dummies.end())
            continue;
          fprintf(stderr, "ERROR: Missing animated mesh %s\n", child_name.c_str());
          return -OBJ_IMPORT_OBJECT_NOT_FOUND;
        }

        subject::mesh_t& mesh = found->second;
        add_mesh(mesh);
      }

      scene->animation->groups[current_group++].n_length = n_current_face - n_group_start_face;
    }

    if (scene->animation)
      scene->animation->n_dynamic_faces = n_current_face - scene->animation->n_static_faces;

    if (n_current_mesh != (int)meshes.size())
    {
      fprintf(stderr, "ERROR: Loaded %d and written %d meshes count mismatch\n", (int)meshes.size(), n_current_mesh);
      return -OBJ_IMPORT_ERROR;
    }

    scene->n_materials = ::max(1, (int)materials.size());
    scene->materials = (subject::material_t*)malloc(sizeof(subject::material_t) * scene->n_materials);
    if (materials.empty())
    {
      scene->materials[0] = subject::black_body();
    }
    else
    {
      memcpy(scene->materials, materials.data(), sizeof(subject::material_t) * scene->n_materials);
    }

    // Generate vertex -> mesh mapping
    {
      std::vector<int> offsets(1, 0);
      for (const auto& vertex_meshes: vertex_to_mesh)
        offsets.push_back(vertex_meshes.size() + offsets.back());
      std::vector<int> index;
      index.reserve(offsets.back());
      for (const auto& vertex_meshes: vertex_to_mesh)
        index.insert(index.end(), vertex_meshes.begin(), vertex_meshes.end());
      scene->n_vertices = vertices.size();
      scene->vertices = (math::vec3*)malloc(vertices.size() * sizeof(math::vec3));
      scene->vertex_to_mesh_offsets = (int*)malloc(offsets.size() * sizeof(int));
      scene->vertex_to_mesh_index = (int*)malloc(index.size() * sizeof(int));
      scene->faces_indices = (int*)malloc(loaded_faces.size() * 3 * sizeof(int));
      memcpy(scene->vertices, vertices.data(), vertices.size() * sizeof(math::vec3));
      memcpy(scene->vertex_to_mesh_offsets, offsets.data(), offsets.size() * sizeof(int));
      memcpy(scene->vertex_to_mesh_index, index.data(), index.size() * sizeof(int));
      memcpy(scene->faces_indices, loaded_faces.data(), loaded_faces.size() * 3 * sizeof(int));
    }

    return OBJ_IMPORT_OK;
  }

  int task(const char* filename
    , int n_meshes
    , const subject::scene_t* scene
    , thermal_solution::task_t* t
    , heat_source_equation::params_t* heat_source
    , parallel_rays_emission_cpu::params_t* distant_source
    , planet_emission::params_t* planet_emission_params
    , subject::task_t* simulation_params
  )
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      fprintf(stderr, "Task file '%s' not found.\n", filename);
      return -OBJ_IMPORT_FILE_ERROR;
    }

    struct action_t
    {
      std::function<bool(const char*)> process;
      std::function<void()> report;
    };

    std::vector<char> mesh_name(1024);
    std::map<std::string, int> mesh_name_to_id;
    std::set<std::string> initialized_meshes;
    if (scene)
    {
      for (int m = 0; m != scene->n_meshes; ++m)
      {
        const subject::mesh_t& mesh = scene->meshes[m];
        if (mesh.name)
          mesh_name_to_id[mesh.name] = m;
      }
    }

    std::map<std::string, action_t> actions;
    if (simulation_params)
    {
      actions["frame_time_scale"] = { [&](const char* line) { return sscanf(line, "%f", &simulation_params->frame_time_scale) == 1; },
      [&]() { fprintf(stderr, "Frame time scale %f\n", simulation_params->frame_time_scale); } };

      actions["simulation_duration"] = { [&](const char* line) { return sscanf(line, "%f", &simulation_params->duration) == 1; },
        [&]() { fprintf(stderr, "Simulation duration %f seconds\n", simulation_params->duration); } };

      actions["solar_rays_count"] = { [&](const char* line) { return sscanf(line, "%d", &simulation_params->n_solar_rays) == 1; },
        [&]() { fprintf(stderr, "Solar (distant) rays count %d\n", simulation_params->n_solar_rays); } };

      actions["emission_rays_count"] = { [&](const char* line) { return sscanf(line, "%d", &simulation_params->n_emission_rays) == 1; },
        [&]() { fprintf(stderr, "Emission rays count %d\n", simulation_params->n_emission_rays); }
      };

      actions["heatmap"] = {
              [&](const char* line) {
                std::vector<char> colormap_name(4096);
                std::vector<char> filename(4096);
                auto success = sscanf(line, "%s %s", colormap_name.data(), filename.data()) == 2;
                simulation_params->heatmap_colormap_name = strdup(colormap_name.data());
                simulation_params->heatmap_filename = strdup(filename.data());
                return success;
              },
              [&]() {
                fprintf(stderr, "Heatmap colormap '%s' to %s\n", simulation_params->heatmap_colormap_name, simulation_params->heatmap_filename);
              }
      };

      actions["tle_orbit"] = { [&](const char* line) {
        std::string library_name = line;
        std::string library_path = library_name;

        if (library_name.find("/\\") > library_name.length())
        {
          std::string base_name(filename);
          size_t dir_pos = base_name.find_last_of("/\\");
          if (dir_pos < base_name.length())
            base_name = base_name.substr(0, dir_pos + 1);
          else
            base_name = "";
          library_path = base_name + library_name;
        }

        fprintf(stderr, "Loading tle orbit '%s'\n", library_path.c_str());

        return orbit(library_path.c_str(), &simulation_params->tle_orbit) == OBJ_IMPORT_OK;
      },
        [&]() {}
      };
    }
    else
    {
      actions["frame_time_scale"] = { [&](const char* line) { return true; },
        [&]() {} };
      actions["simulation_duration"] = { [&](const char* line) { return true; },
        [&]() {} };
      actions["solar_rays_count"] = { [&](const char* line) { return true; },
        [&]() {} };
      actions["emission_rays_count"] = { [&](const char* line) { return true; },
        [&]() {} };
      actions["tle_orbit"] = { [&](const char* line) { return true; },
        [&]() {} };
    }

    if (heat_source)
    {
      actions["htsrc"] = { [&](const char* line) {
        heat_source_equation::heat_source_t source;

        if (sscanf(line, "%s %f", &mesh_name[0], &source.power) != 2)
          return false;
        std::string name(&mesh_name[0]);
        auto found = mesh_name_to_id.find(name);
        if (found == mesh_name_to_id.end())
        {
          fprintf(stderr, "ERROR: Heat source %s not found.\n", name.c_str());
          return false;
        }

        source.mesh_idx = found->second;
        heat_source->sources = (heat_source_equation::heat_source_t*)realloc(heat_source->sources, (1 + heat_source->n_sources) * sizeof(source));
        heat_source->sources[heat_source->n_sources++] = source;
        return true;
      },
        [&]() {} };
    }
    else
    {
      actions["htsrc"] = { [&](const char* line) { return true; },
        [&]() {} };
    }

    if (distant_source)
    {
      actions["dstsrc"] = { [&](const char* line) {
        distant_source->sources = (parallel_rays_emission_cpu::parallel_rays_source_t*)realloc(distant_source->sources, (distant_source->n_sources + 1) * sizeof(parallel_rays_emission_cpu::parallel_rays_source_t));
        parallel_rays_emission_cpu::parallel_rays_source_t& source = distant_source->sources[distant_source->n_sources];
        ++distant_source->n_sources;
        return sscanf(line, "%f %f %f %f", &source.power, &source.direction.x, &source.direction.y, &source.direction.z) == 4;
      },
        [&]() { fprintf(stderr, "Loaded distant rays source with power %f\n", distant_source->sources[distant_source->n_sources - 1].power); }
      };
    }
    else
    {
      actions["dstsrc"] = { [&](const char* line) { return true; },
        [&]() {} };
    }

    if (planet_emission_params)
    {
      actions["emsplnt"] = { [&](const char* line) {
        if (!planet_emission_params)
        {
          fprintf(stderr, "Ignored planet emission '%s'\n", line);
          return true;
        }

        return sscanf(line, "%f %f %f %f %d",
          &planet_emission_params->planet_radius,
          &planet_emission_params->planet_albedo,
          &planet_emission_params->planet_temperature,
          &planet_emission_params->solar_flux,
          &planet_emission_params->n_subdisivion) == 5;
      },
        [&]() { fprintf(stderr, "Loaded planet emission:\n"
          "  radius\t%f\n"
          "  albedo\t%f\n"
          "  planet temper\t%f\n"
          "  solar flux\t%f\n"
          "  n_subdiv\t%d\n",
          planet_emission_params->planet_radius,
          planet_emission_params->planet_albedo,
          planet_emission_params->planet_temperature,
          planet_emission_params->solar_flux,
          planet_emission_params->n_subdisivion
        ); }
      };
    }
    else
    {
      actions["emsplnt"] = { [&](const char* line) { return true; },
        [&]() {} };
    }

    actions["current_time"] = { [&](const char* line) { return sscanf(line, "%lf", &t->current_time) == 1; },
      [&]() { fprintf(stderr, "Current time set to %lf\n", t->current_time); } };

    actions["step"] = { [&](const char* line) { return sscanf(line, "%f", &t->time_delta) == 1; },
      [&]() {fprintf(stderr, "Simulation step %f\n", t->time_delta); } };

    int n_meshes_loaded = 0;
    int allocated = 0;

    float default_tmprt = -1;
    actions["default_tmprt"] = {[&default_tmprt](const char* line) { return sscanf(line, "%f", &default_tmprt) == 1; },
      [&default_tmprt]() {fprintf(stderr, "Default temperature for meshes %f\n", default_tmprt);} };

    actions["tmprt"] = { [&](const char* line) {
        if (allocated <= n_meshes_loaded)
        {
          allocated = n_meshes_loaded + 8;
          t->temperatures = (float*)realloc(t->temperatures, allocated * sizeof(float));
        }

        float temperature;
        if (sscanf(line, "%f %s", &temperature, &mesh_name[0]) == 2)
        {
          std::string name(&mesh_name[0]);
          auto found = mesh_name_to_id.find(name);
          if (found == mesh_name_to_id.end())
          {
            if (mesh_name_to_id.empty())
            {
              // @not This is a corner case, just put an error and retreat.
              fprintf(stderr, "ERROR: Using named meshes for unnamed scene import.\n");
            }
            else
            {
              fprintf(stderr, "ERROR: Mesh '%s' not found during task temperatures import.\n", name.c_str());
            }
            return false;
          }

          int mesh_idx = found->second;
          if (allocated <= mesh_idx)
          {
            allocated = mesh_idx + 8;
            t->temperatures = (float*)realloc(t->temperatures, allocated * sizeof(float));
          }

          ++n_meshes_loaded;
          initialized_meshes.insert(name);
          t->temperatures[mesh_idx] = temperature;
          return true;
        }
        else
        {
          if (!mesh_name_to_id.empty())
          {
            fprintf(stderr, "ERROR: Detected temperature for unnamed mesh during import scene with named meshes.\n");
            return false;
          }

          bool result = sscanf(line, "%f", t->temperatures + (n_meshes_loaded++)) == 1;
          --n_meshes;
          return result;
        }
      },
      [&]() {}
    };

    actions["newfrm"] = { [&](const char*) {
        n_meshes_loaded = 0;
        return true;
      },
      [&]() {}
    };

    std::string line;

    t->current_time = 0;

    while (n_meshes > 0 && getline(file, line))
    {
      if (line == "" || line[0] == '#')
        continue;

      const char* str = line.c_str();
      const char* delim = strchr(str, ' ');
      std::string token = delim ? std::string(str, delim - str) : line;
      const char* params = delim ? delim + 1 : "";

      auto found = actions.find(token);
      if (found == actions.end())
      {
        fprintf(stderr, "Unknown action in line %s.\n", line.c_str());
        continue;
      }

      const action_t& action = found->second;
      if (!action.process(params))
      {
        fprintf(stderr, "Failed to parse line %s.\n", line.c_str());
        return -OBJ_IMPORT_FILE_ERROR;
      }

      action.report();
    }

    if (initialized_meshes.size() != mesh_name_to_id.size())
    {
      if (default_tmprt > 0) {
        fprintf(stderr, "Setting default temperature %f for meshes.'\n", default_tmprt);
        for (const auto& m: mesh_name_to_id)
        {
          if (initialized_meshes.find(m.first) != initialized_meshes.end())
            continue;
          //fprintf(stderr, "%s,", m.first.c_str());
          int mesh_idx = m.second;
          if (allocated <= mesh_idx)
          {
            allocated = mesh_idx + 8;
            t->temperatures = (float*)realloc(t->temperatures, allocated * sizeof(float));
          }

          ++n_meshes_loaded;
          initialized_meshes.insert(m.first);
          t->temperatures[mesh_idx] = default_tmprt;
        }
      }
      else {
        fprintf(stderr, "ERROR: Not all named meshes initialized with its initial temperature value. Broken meshes:\n");
        for (const auto &r : mesh_name_to_id) {
          if (initialized_meshes.find(r.first) == initialized_meshes.end())
            fprintf(stderr, "\t%s\n", r.first.c_str());
        }
        return -OBJ_IMPORT_FILE_ERROR;
      }
    }

    return n_meshes_loaded;
  }

  int task(const char* filename
    , int n_meshes
    , thermal_solution::task_t* t
    , heat_source_equation::params_t* heat_source
    , parallel_rays_emission_cpu::params_t* distant_source
    , planet_emission::params_t* planet_emission_params
    , subject::task_t* simulation_params
  )
  {
    return task(filename, n_meshes, 0, t, heat_source, distant_source, planet_emission_params, simulation_params);
  }

  int task(const char* filename
    , const subject::scene_t* scene
    , thermal_solution::task_t* t
    , heat_source_equation::params_t* heat_source
    , parallel_rays_emission_cpu::params_t* distant_source
    , planet_emission::params_t* planet_emission_params
    , subject::task_t* simulation_params
  )
  {
    return task(filename, scene->n_meshes, scene, t, heat_source, distant_source, planet_emission_params, simulation_params);
  }

  int orbit(const char* filename, ballistics::tle_orbit_t** target)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      fprintf(stderr, "Orbit description file '%s' not found.\n", filename);
      return -OBJ_IMPORT_FILE_ERROR;
    }

    if (!*target)
      *target = (ballistics::tle_orbit_t*)calloc(1, sizeof(ballistics::tle_orbit_t));
    ballistics::tle_orbit_t* description = *target;

    std::string str;
    unsigned line = 0;

    while (getline(file, str))
    {
      ++line;
      if (str == "" || str[0] == '#')
        continue;

      size_t pos = str.find(' ');
      if (pos > str.length())
      {
        fprintf(stderr, "Invalid orbit token %s at line %d.\n", str.c_str(), line);
        return -OBJ_IMPORT_FILE_ERROR;
      }

      std::string token = str.substr(0, pos);
      
      if (token == "orbit_tle_title")
        description->tle.header = strdup(&str[pos + 1]);
      else if (token == "orbit_tle_line_1")
        description->tle.line_1 = strdup(&str[pos + 1]);
      else if (token == "orbit_tle_line_2")
        description->tle.line_2 = strdup(&str[pos + 1]);
      /*
      // @todo Will be enabled with custom attitude axis support
      else if (token == "orbit_attitude_axis")
      {
        char axis = tolower(str[pos + 1]);
        switch (axis)
        {
        case 'x':
        case 'y':
        case 'z':
          description->attitude.type = axis - 'x';
          break;

        default:
          fprintf(stderr, "Invalid orbit attitude axis %s at line %d.\n", str.c_str(), line);
          return -OBJ_IMPORT_FILE_ERROR;
        }
      }*/
      else if (token == "orbit_attitude_type")
      {
        std::string type(&str[pos + 1]);
        for (char& c : type)
          c = tolower(c);
        if (type == "sun")
          description->attitude.type = ATTITUDE_TYPE_SUN;
        else if (type == "nadir")
          description->attitude.type = ATTITUDE_TYPE_NADIR;
        else if (type == "orbit")
          description->attitude.type = ATTITUDE_TYPE_ORBIT;
        else
        {
          fprintf(stderr, "Invalid orbit attitude type %s at line %d.\n", str.c_str(), line);
          return -OBJ_IMPORT_FILE_ERROR;
        }
      }
    }

    return OBJ_IMPORT_OK;
  }
}
