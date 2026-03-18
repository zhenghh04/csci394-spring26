NVCC ?= nvcc
NVFLAGS ?= -O2
LDFLAGS ?=
LDLIBS ?= -lcublas

all: app

app: main.cu
	$(NVCC) $(NVFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f app
