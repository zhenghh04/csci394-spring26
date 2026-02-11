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

all: fib fib_omp fib_omp_dynamic fib_omp_trace fib_omp_dynamic_trace tasks tasks_trace

fib: fib.c
	$(CC) -O2 -Wall -Wextra -o $@ $<

fib_omp: fib_omp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

fib_omp_dynamic: fib_omp_dynamic.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

fib_omp_trace: fib_omp_trace.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

fib_omp_dynamic_trace: fib_omp_dynamic_trace.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

tasks: tasks.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

tasks_trace: tasks_trace.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f fib fib_omp fib_omp_dynamic fib_omp_trace fib_omp_dynamic_trace tasks tasks_trace
