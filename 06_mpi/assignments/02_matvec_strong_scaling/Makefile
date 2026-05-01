CC      = mpicc
CFLAGS  = -O3 -std=c11 -Wall -Wextra

all: main_mpi.x main_series.x

main_mpi.x: main_mpi.c
	$(CC) $(CFLAGS) -o $@ $<

main_series.x: main_series.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f main_mpi.x main_series.x
