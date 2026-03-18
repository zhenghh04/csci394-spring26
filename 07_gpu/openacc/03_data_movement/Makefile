CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
ACCFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

all: app

app: main.c
	$(CC) $(CFLAGS) $(ACCFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f app
