# Makefile Basics

This is a short tutorial for writing and using a simple `Makefile` in this OpenMP module.

## Why use Makefile
- Avoid typing long compile commands repeatedly.
- Keep build commands consistent across machines.
- Add helpful targets such as `all`, `clean`, and `run`.

## Minimal terms
- `target`: something to build (example: `hello_omp`)
- `prerequisite`: file needed to build target (example: `hello_omp.c`)
- `recipe`: command lines run by `make` (must start with a TAB)
- variable: reusable text like compiler flags (`CC`, `CFLAGS`)

## A minimal Makefile (single C file)
```make
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

all: hello_omp

hello_omp: hello_omp.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f hello_omp
```

Notes:
- `$@` means the target name (`hello_omp`).
- `$<` means the first prerequisite (`hello_omp.c`).
- Recipe lines must use a TAB, not spaces.

## Build and run
```bash
make
OMP_NUM_THREADS=4 ./hello_omp
make clean
```

## Add macOS and Linux OpenMP flags
Many folders in this class use this pattern:
```make
UNAME_S := $(shell uname -s)

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=
LDLIBS ?=

ifeq ($(UNAME_S),Darwin)
  CC := clang
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  ifeq ($(OMP_PREFIX),)
    OMP_PREFIX := /opt/homebrew/opt/libomp
    ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
      OMP_PREFIX := /usr/local/opt/libomp
    endif
  endif
  CFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  LDFLAGS += -L$(OMP_PREFIX)/lib
  LDLIBS += -lomp
else
  CFLAGS += -fopenmp
endif
```

Then compile with:
```make
hello_omp: hello_omp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)
```

## Multiple programs in one Makefile
```make
all: a b

a: a.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

b: b.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)
```

## Build executable from `*.o` files
This is the common 2-step build:
1. compile each `.c` into `.o`
2. link all `.o` into one executable

Example:
```make
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp
LDFLAGS ?=
LDLIBS ?=

OBJ = main.o kernel.o utils.o

app: $(OBJ)
	$(CC) -o $@ $(OBJ) $(LDFLAGS) $(LDLIBS)

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

kernel.o: kernel.c
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f app *.o
```

Shorter version with a pattern rule:
```make
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fopenmp

OBJ = main.o kernel.o utils.o

app: $(OBJ)
	$(CC) -o $@ $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f app *.o
```

## Useful quality-of-life targets
```make
.PHONY: all clean run

run: hello_omp
	OMP_NUM_THREADS?=4; ./hello_omp
```

Use `.PHONY` for targets that are actions, not files.

## Common errors
- `missing separator` -> recipe line used spaces instead of TAB.
- `No rule to make target ...` -> filename mismatch in target/prerequisite.
- OpenMP link errors on macOS -> install `libomp` and use include/lib flags.

## Suggested practice
1. In `05_openmp/assignments/`, create your own `Makefile`.
2. Start from the minimal template above.
3. Add `all`, your program target, and `clean`.
4. Test with `make`, `make clean`, and `OMP_NUM_THREADS=4 ./<program>`.
