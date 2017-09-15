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

/**
 * This module contains functionality of loading obj-files (Wavefront).
 */

#include "obj_import.h"
#include "../math/operations.h"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <set>

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

    std::map<std::string, bool> known_unknown_attributes;

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
        if (strncmp("newmtl", str.c_str(), 6) != 0)
        {
          fprintf(stderr, "Failed to parse newmtl at line %d\n", line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        std::string name = str.substr(7);

        subject::material_t m = subject::material_t();

        if (name.length() >= sizeof(m.name))
        {
          fprintf(stderr, "Material name '%s' is too long at line %d\n", name.c_str(), line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        strncpy(m.name, name.c_str(), std::min(sizeof(m.name), name.length()));

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

        while (getline(file, str) && parsedParameters.size() != materialProperties.size())
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
      }
      break;
      }
    }

    return OBJ_IMPORT_OK;
  }

  /// @note Mindflow mode on
  int scene(const char* filename, subject::scene_t** scene)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      fprintf(stderr, "Scene file '%s' not found.\n", filename);
      return -OBJ_IMPORT_FILE_ERROR;
    }

    std::vector<math::vec3> vertices;
    std::vector<idx_face_t> faces;
    std::vector<subject::mesh_t> meshes;
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
        faces.push_back(face);
      }
      break;

      // Ignore shading smoothing group
      case 's':
        continue;

      case 'o':
      case 'g':
      { 
        subject::mesh_t mesh;
        mesh.first_idx = (int)faces.size();
        mesh.material_idx = 0;
        if (!meshes.empty())
          meshes.back().n_faces = (int)(faces.size() - meshes.back().first_idx);
        if (str.length() > 2)
        mesh.name = strdup(str.c_str() + 2);
        meshes.push_back(mesh);
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

          meshes.back().material_idx = found->second;
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

      default:
        fprintf(stderr, "Unknown directive '%s' at line %d\n", str.c_str(), line);
        continue;
        break;
      }
    
    }

    file.close();

    // Preparing scene.
    subject::scene_t* result = subject::scene_create();
    *scene = result;

    // Triangulation to vertices from face indices.
    result->n_faces = (int)faces.size();
    result->faces = (subject::face_t *)malloc(result->n_faces * sizeof(subject::face_t));
    for (int i = 0; i < result->n_faces; ++i)
    {
      idx_face_t idx_face = faces[i];
      result->faces[i] = {
        vertices[idx_face.indices[0]],
        vertices[idx_face.indices[1]],
        vertices[idx_face.indices[2]]
      };
    }

    result->n_meshes = std::max(1, (int)meshes.size());
    result->meshes = (subject::mesh_t *)malloc(sizeof(subject::mesh_t) * result->n_meshes);
    if (meshes.empty())
    {
      result->meshes[0].first_idx = 0;
      result->meshes[0].material_idx = 0;
      result->n_faces = result->meshes[0].n_faces = (int)faces.size();
    }
    else
    {
      meshes.back().n_faces = (int)(faces.size() - meshes.back().first_idx);
      memcpy(result->meshes, meshes.data(), sizeof(subject::mesh_t) * result->n_meshes);
    }

    result->n_materials = std::max(1, (int)materials.size());
    result->materials = (subject::material_t*)malloc(sizeof(subject::material_t) * result->n_materials);
    if (materials.empty())
    {
      result->materials[0] = subject::black_body();
    }
    else
    {
      memcpy(result->materials, materials.data(), sizeof(subject::material_t) * result->n_materials);
    }

    return OBJ_IMPORT_OK;
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

    std::map<std::string, action_t> actions;
    actions["frame_time_scale"] = { [&](const char* line) { return sscanf(line, "%f", &simulation_params->frame_time_scale) == 1; },
      [&]() { fprintf(stderr, "Frame time scale %f\n", simulation_params->frame_time_scale); } };

    actions["current_time"] = { [&](const char* line) { return sscanf(line, "%lf", &t->current_time) == 1; },
      [&]() { fprintf(stderr, "Current time set to %lf\n", t->current_time); } };

    actions["step"] = { [&](const char* line) { return sscanf(line, "%f", &t->time_delta) == 1; },
      [&]() {fprintf(stderr, "Simulation step %f\n", t->time_delta); } };

    actions["simulation_duration"] = { [&](const char* line) { return sscanf(line, "%f", &simulation_params->duration) == 1; },
      [&]() { fprintf(stderr, "Simulation duration %f seconds\n", simulation_params->duration); } };

    actions["solar_rays_count"] = { [&](const char* line) { return sscanf(line, "%d", &simulation_params->n_solar_rays) == 1; },
      [&]() { fprintf(stderr, "Solar (distant) rays count %d\n", simulation_params->n_solar_rays); } };

    actions["emission_rays_count"] = { [&](const char* line) { return sscanf(line, "%d", &simulation_params->n_emission_rays) == 1; },
      [&]() { fprintf(stderr, "Emission rays count %d\n", simulation_params->n_emission_rays); }
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


    int n_meshes_loaded = 0;
    size_t left = 0;

    actions["tmprt"] = { [&](const char* line) {
        if (left == 0)
        {
          left = 8;
          t->temperatures = (float*)realloc(t->temperatures, (n_meshes_loaded + left) * sizeof(float));
        }
        bool result = sscanf(line, "%f", t->temperatures + (n_meshes_loaded++)) == 1;
        --n_meshes;
        --left;
        return result;
      },
      [&]() {}
    };

    actions["newfrm"] = { [&](const char*) {
        left += n_meshes_loaded;
        n_meshes_loaded = 0;
        return true;
      },
      [&]() {}
    };

    actions["htsrc"] = { [&](const char* line) {
        heat_source_equation::heat_source_t source;

        if (sscanf(line, "%d %f", &source.mesh_idx, &source.power) != 2)
          return false;
        heat_source->sources = (heat_source_equation::heat_source_t*)realloc(heat_source->sources, (1 + heat_source->n_sources) * sizeof(source));
        heat_source->sources[heat_source->n_sources++] = source;
        return true;
      },
      [&]() {} };

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
        "  temper\t%f\n"
        "  solar\t%f\n"
        "  n_subdiv\t%d\n",
        planet_emission_params->planet_radius,
        planet_emission_params->planet_albedo,
        planet_emission_params->planet_temperature,
        planet_emission_params->solar_flux,
        planet_emission_params->n_subdisivion
      ); }
    };

    actions["dstsrc"] = { [&](const char* line) {
        distant_source->sources = (parallel_rays_emission_cpu::parallel_rays_source_t*)realloc(distant_source->sources, (distant_source->n_sources + 1) * sizeof(parallel_rays_emission_cpu::parallel_rays_source_t));
        parallel_rays_emission_cpu::parallel_rays_source_t& source = distant_source->sources[distant_source->n_sources];
        ++distant_source->n_sources;
        return sscanf(line, "%f %f %f %f", &source.power, &source.direction.x, &source.direction.y, &source.direction.z) == 4;
      },
      [&]() { fprintf(stderr, "Loaded distant rays source with power %f\n", distant_source->sources[distant_source->n_sources - 1].power); }
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

    return n_meshes_loaded;
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
