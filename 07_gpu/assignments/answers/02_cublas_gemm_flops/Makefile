# cuBLAS FP32 / tensor-core GEMM benchmark
#
# Build:
#   make
#   make CUDA_ARCH=sm_80    # Polaris A100 (default)
#   make CUDA_ARCH=sm_90    # Hopper

NVCC      ?= nvcc
CUDA_ARCH ?= sm_80
NVCC_FLAGS ?= -O2 -arch=$(CUDA_ARCH) -std=c++14
LIBS       ?= -lcublas

SRC = src

all: app_cublas

app_cublas: $(SRC)/app_cublas.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SRC)/app_cublas.cu $(LIBS)

clean:
	rm -f app_cublas

.PHONY: all clean
