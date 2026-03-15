CC := mpicc
CFLAGS := -O3 -std=c11 -Wall -Wextra

all: app app_block

app: main.c
	$(CC) $(CFLAGS) -o $@ $<

app_block: main_block.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f app app_block
