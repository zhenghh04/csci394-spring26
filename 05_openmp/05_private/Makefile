CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: private_example

private_example: private_example.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f private_example
