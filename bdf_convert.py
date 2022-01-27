from collections import defaultdict
import sys


def normalize(s):
    pcs = s.split('.')
    if len(pcs) == 1:
        return s
    if pcs[-1][0] == '+':
        pcs[-1] = 'e' + pcs[-1]
    elif pcs[-1][0] == '-':
        pcs[-1] = 'e' + pcs[-1]
    return '.'.join(pcs)

    
def parse_float(s):
    s = s.strip()
    try:
        return float(s) if s else 0.0
    except ValueError:
        s = normalize(s)
        return float(s)


def parse_int(s):
    s = s.strip()
    return int(s) if s else 0


class Top(object):
    def __init__(self, diffuse=0, specular=0, absorbance=0, transmittance=0, emissivity=0):
        self.diffuse = diffuse
        self.specular = specular
        self.absorbance = absorbance
        self.transmittance = transmittance
        self.emissivity = emissivity


class Material(object):
    def __init__(self, name=None, front=None, rear=None, density=None, thickness=None, heat_capacity=None, thermal_conductivity=None):
        self.name = name
        self.front = front
        self.rear = rear
        self.density = density
        self.heat_capacity = heat_capacity
        self.thickness = thickness
        self.thermal_conductivity = thermal_conductivity

        
class Bdf2Obj(object):
    def __init__(self):
        self.meshes_faces = defaultdict(set)
        self.vertices = dict()
        self.meshes_names = dict()
        self.meshes_materials = dict()
        self.mesh_thickness = dict()
        self.tops = dict()
        self.materials = defaultdict(Material)
        
    def write_parsed(self, obj_filename, mtl_filename):
        self._flatten()
        with open(obj_filename, 'wb') as f:
            f.write(b'mtllib %s\n' % mtl_filename.encode())
            self._write_vertices(f)
            self._write_meshes(f)
        with open(mtl_filename, 'wb') as f:
            self._write_materials(f)            
                
    def _write_vertices(self, f):
        vertices = sorted(self.vertices.items(), key=lambda x: x[0])
        for _idx, v in vertices:
            f.write(b'v %.14f %.14f %.14f\n' % v)
        f.write(b'\n')
    
    def _write_materials(self, f):
        for mesh, material_index in self.meshes_materials.items():        
            name = 'mtl_%s' % self._get_mesh_name(mesh)
            thickness = self.mesh_thickness.get(mesh, float("nan"))
            
            material = self.materials.get(material_index)
            material.name = name
            material.thickness = thickness
            
            self._write_material(f, material)            
    
    def _write_material(self, f, material):
        f.write(b'newmtl %s\n' % material.name.encode())
        f.write(b'shell.density %f\n' % material.density)
        f.write(b'shell.heat_capacity %f\n' % material.heat_capacity)
        f.write(b'shell.thermal_conductivity %f\n' % material.thermal_conductivity)
        f.write(b'shell.thickness %f\n' % material.thickness)
        self._write_top(f, b'front', material.front)
        self._write_top(f, b'rear', material.rear)
        f.write(b'\n')
            
    def _write_top(self, f, side, top):
        f.write(b'%s.specular_reflectance %f\n' % (side, top.specular))
        f.write(b'%s.diffuse_reflectance %f\n' % (side, top.diffuse))
        f.write(b'%s.absorbance %f\n' % (side, top.absorbance))
        f.write(b'%s.transmittance %f\n' % (side, top.transmittance))
        f.write(b'%s.emissivity %f\n' % (side, top.emissivity))
        
    def _get_mesh_name(self, index):
        return self.meshes_names.get(index, 'mesh_%d' % index)
    
    def _write_meshes(self, f):
        for mesh, faces in self.meshes_faces.items():
            name = self._get_mesh_name(mesh)
            f.write(b'g %s\n' % name.encode())
            
            material = self.meshes_materials.get(mesh)            
            if material:
                material_name = 'mtl_%s' % name
                f.write(b'usemtl %s\n' % material_name.encode())
            
            sorted_faces = sorted(faces)            
            for v in sorted_faces:
                f.write(b'f %d %d %d\n' % v)
            f.write(b'\n')
                
    def _flatten(self):
        remap = dict()
        for index, _ in self.vertices.items():
            new_index = 1 + len(remap)
            remap[index] = new_index
        new_meshes = dict()                
        for mesh, faces in self.meshes_faces.items():
            reindexed_faces = set()
            for v0, v1, v2 in faces:
                reindexed_face = (remap[v0], remap[v1], remap[v2])
                reindexed_faces.add(reindexed_face)
            new_meshes[mesh] = reindexed_faces
        self.meshes_faces = new_meshes
            

    def validate(self):
        for mesh, faces in self.meshes_faces.items():
            for v0, v1, v2 in faces:
                if v0 not in self.vertices:
                    raise KeyError('Missing mesh %d vertice %d' % (mesh, v0))
                if v1 not in self.vertices:
                    raise KeyError('Missing mesh %d vertice %d' % (mesh, v1))
                if v2 not in self.vertices:
                    raise KeyError('Missing mesh %d vertice %d' % (mesh, v2))
                        

    def parse_top(self, filename):
        with open(filename) as f:
            lines = [x for x in f.read().split('\n')]
            self._parse_top_lines(lines)
    
    def _parse_top_lines(self, lines):
        medium_line = None
        for line in lines:
            if medium_line:
                self.parse_medium1(medium_line, line)
                medium_line = None
            elif line.startswith('FTOP'):
                self.parse_ftop(line)
            elif line.startswith('MEDIUM1'):
                medium_line = line
                
    def parse_medium1(self, l0, l1):
        index = parse_int(l0[8:16])
        thermal_conductivity = parse_float(l0[32:40])
        heat_capacity = parse_float(l0[40:48])
        front = self.tops.get(parse_int(l1[8:16]))
        rear = self.tops.get(parse_int(l1[16:24]))
        
        material = self.materials[index]
        material.front = front
        material.rear = rear
        material.thermal_conductivity = thermal_conductivity
        material.heat_capacity = heat_capacity
                
    def parse_ftop(self, line):
        index = parse_int(line[8:16])
        diffuse = parse_float(line[16:24])
        specular = parse_float(line[24:32])
        absorbance = parse_float(line[32:40])
        transmittance = parse_float(line[40:48])        
        emissivity = parse_float(line[48:56])
        
        top = Top(
            diffuse=diffuse,
            specular=specular,
            absorbance=absorbance,
            transmittance=transmittance,
            emissivity=emissivity,
        )
        self.tops[index] = top
        
        
    def parse_bdf(self, filename):
        with open(filename) as f:
            lines = [x for x in f.read().split('\n')]
            self._parse_bdf_lines(lines)
    
    def _parse_bdf_lines(self, lines):
        grid_line = None
        active_mesh = None
        active_material_name = None
        for line in lines:
            if active_mesh:
                if line.startswith('$ Pset: '):
                    name = line.split(' ')[2][1:-1]
                    print('Set mesh %d name %s' % (active_mesh, name))
                    self.meshes_names[active_mesh] = name
                active_mesh = None
            elif line.startswith('$ Material Record'):
                active_material_name = line.split(' ')[4]
            elif grid_line:
                self.parse_grid(grid_line, line)
                grid_line = None
            elif line.startswith('CQUAD4'):
                self.parse_cquad4(line)
            elif line.startswith('CTRIA3'):
                self.parse_ctria3(line)
            elif line.startswith('GRID*'):
                grid_line = line
                continue
            elif line.startswith('PSHELL'):
                active_mesh = self.parse_pshell(line)
            elif line.startswith('MAT1'):
                index = self.parse_mat1(line, active_material_name)
                active_material_name = None
                
    def parse_mat1(self, line, name):
        index = parse_int(line[9:16])
        density = parse_float(line[40:48])
        material = self.materials[index]
        if name:
            print('Set material %d name %s' % (index, name))
        material.name = name
        material.density = density
    
    def parse_pshell(self, line):
        index = parse_int(line[9:17])
        material = parse_int(line[17:24])
        thickness = parse_float(line[24:32])
        self.mesh_thickness[index] = thickness
        self.meshes_materials[index] = material
        return index
                
    def parse_grid(self, l0, l1):
        index = parse_int(l0[9:17])
        x = parse_float(l0[40:56])
        y = parse_float(l0[56:72])
        z = parse_float(l1[7:24])
        self.vertices[index] = (x, y, z)
                
    def parse_cquad4(self, line):
        index = parse_int(line[9:17])
        mesh = parse_int(line[17:25])
        vert0 = parse_int(line[25:33])
        vert1 = parse_int(line[33:41])
        vert2 = parse_int(line[41:49])
        vert3 = parse_int(line[49:57])

        face0 = (vert0, vert1, vert2)
        face1 = (vert0, vert2, vert3)
        self._add_mesh_face(mesh, face0)
        self._add_mesh_face(mesh, face1)
        
    def parse_ctria3(self, line):
        index = parse_int(line[9:17])
        mesh = parse_int(line[17:25])
        vert0 = parse_int(line[25:33])
        vert1 = parse_int(line[33:41])
        vert2 = parse_int(line[41:49])
        
        face0 = (vert0, vert1, vert2)
        self._add_mesh_face(mesh, face0)
        
    def _add_mesh_face(self, mesh, face):
        #print('Add mesh %d face' % mesh, face)
        self.meshes_faces[mesh].add(face)


def main():
    bdf = sys.argv[1]
    top = sys.argv[2]
    obj = sys.argv[3]
    mtl = sys.argv[4]
    parser = Bdf2Obj()
    parser.parse_bdf(bdf)
    parser.parse_top(top)
    parser.validate()
    parser.write_parsed(obj, mtl)


if __name__ == "__main__":
    main()

