NVCC     ?= nvcc
NVFLAGS  ?= -O2
CUDA_ARCH ?= sm_80   # A100 on Polaris

TARGET = matmul

all: $(TARGET)

$(TARGET): matmul.cu
	$(NVCC) $(NVFLAGS) -arch=$(CUDA_ARCH) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
