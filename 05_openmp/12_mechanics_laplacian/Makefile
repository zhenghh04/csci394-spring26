UNAME_S := $(shell uname -s)

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

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
CFLAGS += -fopenmp
endif

all: laplacian2d

laplacian2d: laplacian2d.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f laplacian2d
