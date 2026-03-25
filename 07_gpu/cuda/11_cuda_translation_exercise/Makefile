NVCC ?= nvcc
NVFLAGS ?= -O2
CC ?= cc
CFLAGS ?= -O2
CUDA_ARCH ?=
COMMON_GENCODE ?= \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90
LDFLAGS ?=
LDLIBS ?=

ifeq ($(CUDA_ARCH),)
GPUFLAGS := $(COMMON_GENCODE)
else
GPUFLAGS := -arch=$(CUDA_ARCH)
endif

all: app_cpu app_solution

app: main.cu
	$(NVCC) $(NVFLAGS) $(GPUFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

app_solution: solution.cu
	$(NVCC) $(NVFLAGS) $(GPUFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

app_cpu: main.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f app app_cpu app_solution
