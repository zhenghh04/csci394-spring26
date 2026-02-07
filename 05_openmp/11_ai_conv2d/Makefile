CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: conv2d

conv2d: conv2d.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f conv2d
