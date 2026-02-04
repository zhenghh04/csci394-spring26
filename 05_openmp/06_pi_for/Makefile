CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: pi_omp

pi_omp: pi_omp.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f pi_omp
