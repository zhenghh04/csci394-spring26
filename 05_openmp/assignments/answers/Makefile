UNAME_S := $(shell uname -s)

CC      ?= gcc
CFLAGS  ?= -O3 -Wall -Wextra -std=c11
LDFLAGS ?=
LDLIBS  ?=

ifeq ($(UNAME_S),Darwin)
  # macOS: clang needs Homebrew libomp
  CC := clang
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  ifeq ($(OMP_PREFIX),)
    OMP_PREFIX := /opt/homebrew/opt/libomp
    ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
      OMP_PREFIX := /usr/local/opt/libomp
    endif
  endif
  CFLAGS  += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  LDFLAGS += -L$(OMP_PREFIX)/lib
  LDLIBS  += -lomp
else
  # Linux (Crux: gcc with -fopenmp)
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
endif

all: loopy_assignment

loopy_assignment: loopy_assignment.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f loopy_assignment *.o *.dat timings.csv

.PHONY: all clean
