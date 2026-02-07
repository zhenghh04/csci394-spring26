CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: heat2d

heat2d: heat2d.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f heat2d
