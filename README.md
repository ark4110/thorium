# Welcome to Thorium engine!

Thorium is a set of libraries and tools to solve radiance and conductive thermal problems.

Features:

* Heat solution for conductive and radiance problems.
* Implement correct physics of the thermal problem.
* Implement all basic physical and optical properties of materials: density, heat capacity, thermal conductivity, emissivity, specular/diffuse reflectance, absorbance and transmittance.
* Use monte-carlo method for radiance problem simulation.
* Support elements self-emission radiance, direct Sun, Sun reflected from Earth and Earth self radiance.
* Support static heat source/sink elements.
* Implement affine transformation of geometry.
* Use [Blender 3D editor](https://www.blender.org/) as front-end for problem definition and results visualization.
* Support satellite orbital orientation and ballistic definition using TLE.
* Use simple [OBJ format](https://ru.wikipedia.org/wiki/Obj) for geometry and problem definition.
* Performant. Calculate up to 20 integration steps per second for problem with 200K triangles using 10M simulation rays.
* Scales up to 60M simulation rays and millions of triangles on GTX 1080.
* Runs on Windows and Linux. Written in C/C++and CUDA.
* Provide C API for programming language bindings generation.


Upcoming features:

* Multi-GPU support with linear scaling by GPU count.
* Generation of rays with different frequency depending on source temperature.
* Nonlinear optical properties dependent on frequency.

Limitations:

* Only shell (2D thin constructs) elements supported.
* Conductive solution support is limited to 2D shell and triangle elements.
* Only triangle supported as basic geometry element (i.e. no quads or bars).


[Installation instructions](https://bitbucket.org/radiosity/engine/wiki/Install)

[Building instructions](https://bitbucket.org/radiosity/engine/wiki/Build)

[Architecture](https://bitbucket.org/radiosity/engine/wiki/Architecture)
