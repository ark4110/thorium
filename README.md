# Welcome

Welcome to Thorium engine!
Thorium is a set of libraries and tools used to solve radiance and conductive thermal problems.


## Running on Windows
Download repository and go /bin directory. There are 64-bit version of Thorium binaries and corresponding .dll

[Visual Studio 2015 Redistributable](https://www.microsoft.com/download/details.aspx?id=48145) may be required.


## Building on Windows

1. Install [Visual Studio 2015](https://www.visualstudio.com/downloads/). Community Edition is enough.

2. Install Nvidia [CUDA](https://developer.nvidia.com/cuda-downloads). Nvidia developer program registration is required for download.

3. Install Nvidia [OptiX](https://developer.nvidia.com/designworks/optix/download). Define OPTIX_INCLUDE_DIR and OPTIX_LIB_X64_DIR environment variables. Default installation directory for OptiX is C:\ProgramData\NVIDIA Corporation\OptiX SDK 4.0.1

4. Open solution.sln and build.

Please notice that due to OptiX 64-bit only distribution, most tools has only 64-bit version or will be compiled without OptiX ray caster support.


## Building on Linux

Required components:

1. gcc and tool chain.

2. cmake >3.0.

3. Nvidia [CUDA](https://developer.nvidia.com/cuda-downloads)

Building:

1. Optix for Linux is already included into distribution. (Should be removed in #7)

2. Create out-of-source build directory.

3. cmake ../engine

4. make

5. Binaries will be deployed to bin directory