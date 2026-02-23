CC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

APPS := bcast scatter gather reduce

all: $(APPS)

bcast: bcast.c
	$(CC) $(CFLAGS) -o $@ $<

scatter: scatter.c
	$(CC) $(CFLAGS) -o $@ $<

gather: gather.c
	$(CC) $(CFLAGS) -o $@ $<

reduce: reduce.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(APPS)
