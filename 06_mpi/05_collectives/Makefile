CC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra

all: app

app: main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f app
