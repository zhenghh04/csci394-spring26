CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: hello_omp

hello_omp: hello_omp.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f hello_omp
