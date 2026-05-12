CXX ?= c++
NVCC ?= nvcc
ARCH ?= sm_86
BUILD_DIR ?= build
CXXFLAGS ?= -O2 -std=c++17 -Iinclude
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -Iinclude

CPU_SRCS := src/cpu_reference.cpp
CUDA_SRCS := src/quantize.cu src/matmul.cu src/activation.cu src/inference_engine.cu

.PHONY: all cpu-test inference clean

all: cpu-test inference

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

cpu-test: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPU_SRCS) tests/test_cpu_reference.cpp -o $(BUILD_DIR)/cpu_reference_test

inference: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRCS) src/main.cu src/cpu_reference.cpp -o $(BUILD_DIR)/inference

clean:
	rm -rf $(BUILD_DIR)
