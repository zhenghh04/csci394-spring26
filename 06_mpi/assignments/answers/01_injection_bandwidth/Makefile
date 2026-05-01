CC      = mpicc
CFLAGS  = -O3 -std=c11 -Wall -Wextra

all: app_injection

app_injection: main_injection.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f app_injection
