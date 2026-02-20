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

all: atomic_counter critical_counter barrier_demo single_master_demo

atomic_counter: atomic_counter.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

critical_counter: critical_counter.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

barrier_demo: barrier_demo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

single_master_demo: single_master_demo.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

counter: counter.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f atomic_counter critical_counter barrier_demo single_master_demo
