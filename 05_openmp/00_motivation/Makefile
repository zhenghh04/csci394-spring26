CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: motivation

motivation: motivation.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f motivation
