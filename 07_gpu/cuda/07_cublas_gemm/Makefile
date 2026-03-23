NVCC ?= nvcc
NVFLAGS ?= -O2
CUDA_ARCH ?=
COMMON_GENCODE ?= \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_86,code=sm_86 \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90
LDFLAGS ?=
LDLIBS ?= -lcublas

ifeq ($(CUDA_ARCH),)
GPUFLAGS := $(COMMON_GENCODE)
else
GPUFLAGS := -arch=$(CUDA_ARCH)
endif

all: app

app: main.cu
	$(NVCC) $(NVFLAGS) $(GPUFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f app
