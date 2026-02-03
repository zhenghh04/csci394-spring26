CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: reduction_timing

reduction_timing: reduction_timing.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f reduction_timing
