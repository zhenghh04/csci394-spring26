CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp -lm

all: lj_forces

lj_forces: lj_forces.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f lj_forces
