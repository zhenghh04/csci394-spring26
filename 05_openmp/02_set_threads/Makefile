CC = gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: set_threads

set_threads: set_threads.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f set_threads
