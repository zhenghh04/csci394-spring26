CC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

all: io_test

io_test: io_test.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f io_test
