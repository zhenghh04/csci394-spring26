CC = mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

all: pingpong

pingpong: pingpong.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f pingpong
