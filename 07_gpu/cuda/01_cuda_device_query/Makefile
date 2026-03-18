NVCC ?= nvcc
NVFLAGS ?= -O2
LDFLAGS ?=
LDLIBS ?=

all: app

app: main.cu
	$(NVCC) $(NVFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f app
