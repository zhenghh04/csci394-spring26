# GPU offload comparison Makefile
#
# Polaris (NVHPC compiler `nvc` via `cc` wrapper):
#   make all
# OR explicitly:
#   make CC=nvc CXX=nvc++ NVCC=nvcc CUDA_ARCH=cc80
#
# Targets:
#   cpu          - OpenMP CPU baseline (compiler must support OpenMP)
#   openacc      - OpenACC offload (requires nvc)
#   omp_target   - OpenMP target offload (requires nvc -mp=gpu)
#   cuda         - CUDA kernel via nvcc
#   pytorch      - just a marker; runs `python src/app_pytorch.py`

CC          ?= nvc
NVCC        ?= nvcc
CUDA_ARCH   ?= cc80
SM          ?= sm_80

# Polaris NVHPC uses -mp=gpu / -acc; -fast for SIMD on host
CPU_FLAGS    ?= -O2 -mp -fast
ACC_FLAGS    ?= -O2 -acc -gpu=$(CUDA_ARCH) -Minfo=accel
OMPT_FLAGS   ?= -O2 -mp=gpu -gpu=$(CUDA_ARCH) -Minfo=mp
NVCC_FLAGS   ?= -O2 -arch=$(SM) -std=c++14

SRC = src

all: cpu openacc omp_target cuda

cpu: app_cpu
openacc: app_openacc
omp_target: app_omp_target
cuda: app_cuda

app_cpu: $(SRC)/app_cpu.c $(SRC)/common.h
	$(CC) $(CPU_FLAGS) -o $@ $(SRC)/app_cpu.c

app_openacc: $(SRC)/app_openacc.c $(SRC)/common.h
	$(CC) $(ACC_FLAGS) -o $@ $(SRC)/app_openacc.c

app_omp_target: $(SRC)/app_omp_target.c $(SRC)/common.h
	$(CC) $(OMPT_FLAGS) -o $@ $(SRC)/app_omp_target.c

app_cuda: $(SRC)/app_cuda.cu $(SRC)/common.h
	$(NVCC) $(NVCC_FLAGS) -I$(SRC) -o $@ $(SRC)/app_cuda.cu

# Convenience target — pytorch needs only python; print run command
pytorch:
	@echo "Run: python $(SRC)/app_pytorch.py 1024 5 1"

clean:
	rm -f app_cpu app_openacc app_omp_target app_cuda

.PHONY: all cpu openacc omp_target cuda pytorch clean
