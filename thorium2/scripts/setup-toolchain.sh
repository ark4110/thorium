#!/usr/bin/env bash
# Installs a C++ toolchain without root (micromamba + conda-forge) into $PREFIX
# (default /tmp/tools/dev): GCC, CMake, Ninja, Eigen 3.4, GoogleTest.
# Then: . scripts/env.sh
set -euo pipefail
PREFIX=${PREFIX:-/tmp/tools/dev}
MAMBA_DIR=${MAMBA_DIR:-/tmp/tools/mamba}
mkdir -p "$MAMBA_DIR"
cd "$MAMBA_DIR"
if [ ! -x bin/micromamba ]; then
  curl -sL -o mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/latest
  python3 -c "import tarfile; tarfile.open('mm.tar.bz2').extract('bin/micromamba', '.')"
fi
export MAMBA_ROOT_PREFIX="$MAMBA_DIR/root"
./bin/micromamba create -y -q -p "$PREFIX" -c conda-forge cxx-compiler cmake ninja eigen=3.4 gtest make
echo "Toolchain in $PREFIX. Run: . scripts/env.sh"
