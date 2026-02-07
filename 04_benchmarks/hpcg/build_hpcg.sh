#!/bin/bash
set -euo pipefail

HPCG_VERSION=${HPCG_VERSION:-3.1}
HPCG_TARBALL=${HPCG_TARBALL:-hpcg-${HPCG_VERSION}.tar.gz}
HPCG_DIR=${HPCG_DIR:-hpcg-${HPCG_VERSION}}
HPCG_URL=${HPCG_URL:-https://www.hpcg-benchmark.org/downloads/${HPCG_TARBALL}}

# MPI compiler wrappers
MPICC=${MPICC:-mpicc}
MPICXX=${MPICXX:-mpicxx}

# Build target name inside the HPCG source tree
ARCH=${ARCH:-Linux}

if [ ! -f "$HPCG_TARBALL" ]; then
  echo "Downloading ${HPCG_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$HPCG_TARBALL" "$HPCG_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$HPCG_TARBALL" "$HPCG_URL"
  else
    echo "Neither curl nor wget found" >&2
    exit 1
  fi
fi

if [ ! -d "$HPCG_DIR" ]; then
  tar -xzf "$HPCG_TARBALL"
fi

# Create a minimal setup for the chosen architecture
cat > "${HPCG_DIR}/setup/Make.${ARCH}" <<EOM
# Minimal HPCG build config (edit for your system)
CXX = ${MPICXX}
CC  = ${MPICC}
LINKER = ${MPICXX}

CXXFLAGS = -O3 -std=c++11
LINKFLAGS =

# You may need to add BLAS or other libs here
LIBS =
EOM

# Build
make -C "${HPCG_DIR}" arch=${ARCH}

# Print binary location
if [ -f "${HPCG_DIR}/bin/${ARCH}/xhpcg" ]; then
  echo "Built: ${HPCG_DIR}/bin/${ARCH}/xhpcg"
fi
