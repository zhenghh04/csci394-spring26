UNAME_S := $(shell uname -s)

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=
OFFLOAD_FLAGS ?=

ifeq ($(UNAME_S),Darwin)
CC := clang
OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifeq ($(OMP_PREFIX),)
OMP_PREFIX := /opt/homebrew/opt/libomp
endif
CFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
LDFLAGS += -L$(OMP_PREFIX)/lib
LDLIBS += -lomp
else
CFLAGS += -fopenmp
endif

all: app

app: main.c
	$(CC) $(CFLAGS) $(OFFLOAD_FLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f app
