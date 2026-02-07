#!/bin/bash
set -euo pipefail

HPL_VERSION=${HPL_VERSION:-2.3}
HPL_TARBALL=${HPL_TARBALL:-hpl-${HPL_VERSION}.tar.gz}
HPL_DIR=${HPL_DIR:-hpl-${HPL_VERSION}}
HPL_URL=${HPL_URL:-https://www.netlib.org/benchmark/hpl/${HPL_TARBALL}}
ARCH=${ARCH:-UNKNOWN}
CC=${CC:-mpicc}
LINKER=${LINKER:-mpicc}
BLAS_LIBS=${BLAS_LIBS:--lopenblas}

if [ ! -f "$HPL_TARBALL" ]; then
  echo "Downloading ${HPL_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$HPL_TARBALL" "$HPL_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$HPL_TARBALL" "$HPL_URL"
  else
    echo "Neither curl nor wget found" >&2
    exit 1
  fi
fi

if [ ! -d "$HPL_DIR" ]; then
  tar -xzf "$HPL_TARBALL"
fi

cat > "Make.${ARCH}" <<EOM
SHELL        = /bin/sh
CD           = cd
CP           = cp
LN_S         = ln -s
MKDIR        = mkdir -p
RM           = /bin/rm -f

ARCH         = ${ARCH}
TOPdir       = $(pwd)/${HPL_DIR}
INCdir       = \\$(TOPdir)/include
BINdir       = \\$(TOPdir)/bin/\\$(ARCH)
LIBdir       = \\$(TOPdir)/lib/\\$(ARCH)
HPLlib       = \\$(LIBdir)/libhpl.a

MPdir        =
MPinc        =
MPlib        =

LAdir        =
LAinc        =
LAlib        = ${BLAS_LIBS}

CC           = ${CC}
CCNOOPT      = ${CC}
LINKER       = ${LINKER}
LINKFLAGS    =

HPL_OPTS     =
HPL_DEFS     =

CCFLAGS      = \\$(HPL_OPTS) \\$(HPL_DEFS) -O3 -fomit-frame-pointer
LINKFLAGS    = \\$(CCFLAGS)

HPL_INCLUDES = \\$(MPinc) \\$(LAinc) \\$(INCdir)
HPL_LIBS     = \\$(HPLlib) \\$(LAlib) \\$(MPlib)

F2CDEFS      =
F2CINC       =
F2CLIBS      =
EOM

cp "Make.${ARCH}" "${HPL_DIR}/Make.${ARCH}"
make -C "${HPL_DIR}" arch="${ARCH}"
