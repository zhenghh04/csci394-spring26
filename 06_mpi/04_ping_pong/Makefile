CC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

all: app app_pairs

app: main.c
	$(CC) $(CFLAGS) -o $@ $<

app_pairs: main_pairs.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f app app_pairs
