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

all: parallel_for_serial parallel_for_omp parallel_for_dist parallel_for_static_chunk parallel_for_dynamic_chunk parallel_for_schedule_bench

parallel_for_serial: parallel_for_serial.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

parallel_for_omp: parallel_for_omp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

parallel_for_dist: parallel_for_dist.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

parallel_for_static_chunk: parallel_for_static_chunk.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

parallel_for_dynamic_chunk: parallel_for_dynamic_chunk.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

parallel_for_schedule_bench: parallel_for_schedule_bench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f parallel_for_serial parallel_for_omp parallel_for_dist parallel_for_static_chunk parallel_for_dynamic_chunk parallel_for_schedule_bench
