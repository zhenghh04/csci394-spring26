# HPL build helper (template). Adjust paths and libraries for your cluster.

HPL_VERSION ?= 2.3
HPL_TARBALL ?= hpl-$(HPL_VERSION).tar.gz
HPL_DIR ?= hpl-$(HPL_VERSION)
HPL_URL ?= https://www.netlib.org/benchmark/hpl/$(HPL_TARBALL)

# Compiler + MPI
CC ?= mpicc
LINKER ?= mpicc

# BLAS/LAPACK (adjust to your system)
# Examples:
#   BLAS_LIBS=-lopenblas
#   BLAS_LIBS=-L/path/to/mkl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
BLAS_LIBS ?= -lopenblas

# Build architecture tag used by HPL (creates Make.UNKNOWN)
ARCH ?= UNKNOWN

all: build

# Download the HPL tarball
$(HPL_TARBALL):
	@echo "Downloading $(HPL_URL)"
	@command -v curl >/dev/null 2>&1 && curl -L -o $@ $(HPL_URL) || \
	 (command -v wget >/dev/null 2>&1 && wget -O $@ $(HPL_URL)) || \
	 (echo "Neither curl nor wget found" && exit 1)

# Extract source
$(HPL_DIR): $(HPL_TARBALL)
	tar -xzf $(HPL_TARBALL)

# Create a minimal Makefile include for HPL
Make.$(ARCH): | $(HPL_DIR)
	@echo "Creating Make.$(ARCH)";
	@cat > Make.$(ARCH) <<'EOM'
SHELL        = /bin/sh
CD           = cd
CP           = cp
LN_S         = ln -s
MKDIR        = mkdir -p
RM           = /bin/rm -f

ARCH         = $(ARCH)
TOPdir       = $(PWD)/$(HPL_DIR)
INCdir       = $(TOPdir)/include
BINdir       = $(TOPdir)/bin/$(ARCH)
LIBdir       = $(TOPdir)/lib/$(ARCH)
HPLlib       = $(LIBdir)/libhpl.a

MPdir        =
MPinc        =
MPlib        =

LAdir        =
LAinc        =
LAlib        = $(BLAS_LIBS)

CC           = $(CC)
CCNOOPT      = $(CC)
LINKER       = $(LINKER)
LINKFLAGS    =

HPL_OPTS     =
HPL_DEFS     =

# Basic flags (adjust as needed)
CCFLAGS      = $(HPL_OPTS) $(HPL_DEFS) -O3 -fomit-frame-pointer
LINKFLAGS    = $(CCFLAGS)

# MPI
HPL_INCLUDES = $(MPinc) $(LAinc) $(INCdir)
HPL_LIBS     = $(HPLlib) $(LAlib) $(MPlib)

# BLAS/LAPACK
F2CDEFS      =
F2CINC       =
F2CLIBS      =
EOM

# Build HPL using the generated Makefile
build: $(HPL_DIR) Make.$(ARCH)
	@cp Make.$(ARCH) $(HPL_DIR)/Make.$(ARCH)
	$(MAKE) -C $(HPL_DIR) arch=$(ARCH)

clean:
	$(MAKE) -C $(HPL_DIR) arch=$(ARCH) clean || true

distclean:
	$(RM) -r $(HPL_DIR) $(HPL_TARBALL) Make.$(ARCH)

.PHONY: all build clean distclean
