set -euo pipefail

wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar -xzf hpl-2.3.tar.gz
cd hpl-2.3

# Allow callers to override BLAS/LAPACK flags, and honor OPENBLASROOT if set.
BLAS_LIBS="${BLAS_LIBS:-"-lopenblas"}"
CPPFLAGS="${CPPFLAGS:-}"
LDFLAGS="${LDFLAGS:-}"

if [[ -n "${OPENBLASROOT:-}" ]]; then
  CPPFLAGS="${CPPFLAGS} -I${OPENBLASROOT}/include"
  LDFLAGS="${LDFLAGS} -L${OPENBLASROOT}/lib"
fi

./configure --prefix="$PWD/build" CC=mpicc \
  CPPFLAGS="${CPPFLAGS}" \
  LDFLAGS="${LDFLAGS}" \
  BLAS_LIBS="${BLAS_LIBS}"

make -j 8 install
