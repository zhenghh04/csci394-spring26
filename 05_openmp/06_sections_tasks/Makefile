CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: sections_tasks

sections_tasks: sections_tasks.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f sections_tasks
