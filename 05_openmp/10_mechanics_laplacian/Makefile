CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: laplacian2d

laplacian2d: laplacian2d.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f laplacian2d
