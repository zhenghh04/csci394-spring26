UNAME_S := $(shell uname -s)

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=
CC_VERSION_LINE := $(shell $(CC) --version 2>/dev/null | head -n 1)
IS_NVHPC := $(findstring NVIDIA,$(CC_VERSION_LINE))

# Optional target-offload flags (set from command line if needed)
# Example (LLVM+NVIDIA):
# make OFFLOAD_FLAGS='-fopenmp-targets=nvptx64-nvidia-cuda'
OFFLOAD_FLAGS ?=

ifeq ($(UNAME_S),Darwin)
  CC := clang
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  ifeq ($(OMP_PREFIX),)
    OMP_PREFIX := /opt/homebrew/opt/libomp
    ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
      OMP_PREFIX := /usr/local/opt/libomp
    endif
  endif
  CFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  LDFLAGS += -L$(OMP_PREFIX)/lib
  LDLIBS += -lomp
else
ifneq ($(IS_NVHPC),)
  # Polaris with NVIDIA HPC compiler (cc -> nvc): OpenMP GPU offload flags.
  CFLAGS += -mp=gpu -gpu=cc80
else
  CFLAGS += -fopenmp
endif
endif

all: target_axpy parallel_axpy

target_axpy: target_axpy.c
	$(CC) $(CFLAGS) $(OFFLOAD_FLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

parallel_axpy: parallel_axpy.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f target_axpy parallel_axpy
