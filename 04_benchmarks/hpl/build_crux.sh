set -euo pipefail
module load cray-libsci

wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar -xzf hpl-2.3.tar.gz
cd hpl-2.3

# Allow callers to override BLAS/LAPACK flags, and honor CRAY_LIBSCI_PREFIX if set.
# On Cray, LibSci provides BLAS/LAPACK.
BLAS_LIBS="${BLAS_LIBS:-"-lsci_cray"}"
CPPFLAGS="${CPPFLAGS:-}"
LDFLAGS="${LDFLAGS:-}"

if [[ -n "${CRAY_LIBSCI_PREFIX:-}" ]]; then
  LDFLAGS="${LDFLAGS} -L${CRAY_LIBSCI_PREFIX}/lib"
fi

# HPL's configure script doesn't know LibSci, so add it to the probe list.
if [[ -f configure ]] && ! grep -q "Cray LibSci" configure; then
  awk -v blas="${BLAS_LIBS}" '
    /^libs10=-lblas$/ {
      print
      print ""
      print "name11=Cray LibSci"
      print "rout11=dgemm_"
      print "libs11=" blas
      next
    }
    { print }
  ' configure > configure.tmp && mv configure.tmp configure

  sed -i 's/for hpl_i in 1 2 3 4 5 6 7 8 9 10;/for hpl_i in 1 2 3 4 5 6 7 8 9 10 11;/' configure
fi

chmod +x configure
./configure --prefix="$PWD/build" CC=mpicc \
  CPPFLAGS="${CPPFLAGS}" \
  LDFLAGS="${LDFLAGS}" \
  BLAS_LIBS="${BLAS_LIBS}"

make -j 8 install
