#Makefile
#!/bin/bash
CC = gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp   

all: parallel_for_serial parallel_for_omp parallel_for_dist

parallel_for_serial: parallel_for_serial.c
	$(CC) $(CFLAGS) -o $@ $<

parallel_for_omp: parallel_for_omp.c
	$(CC) $(CFLAGS) -o $@ $<

parallel_for_dist: parallel_for_dist.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f parallel_for_serial parallel_for_omp parallel_for_dist
