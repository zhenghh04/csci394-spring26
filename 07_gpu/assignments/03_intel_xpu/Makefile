# Aurora Intel XPU GEMM (FP32 + XMX joint_matrix)
#
# Build (Aurora):
#   module load oneapi
#   make
#
# Targets:
#   app_xpu  - combined fp32 / xmx_bf16_fp32 binary

CXX        ?= icpx
SYCLFLAGS  ?= -fsycl -O2 -fsycl-targets=spir64
EXTRAFLAGS ?=

SRC = src

all: app_xpu

app_xpu: $(SRC)/app_xpu.cpp
	$(CXX) $(SYCLFLAGS) $(EXTRAFLAGS) -o $@ $(SRC)/app_xpu.cpp

clean:
	rm -f app_xpu

.PHONY: all clean
