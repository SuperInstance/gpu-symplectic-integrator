CUDA_PATH ?= /usr/local/cuda-12.6
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = -arch=sm_89 -O2 -Iinclude
LIBS = -lcudart

SRCDIR = src
OBJDIR = obj

SRCS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRCS))

.PHONY: all test bench clean lib

all: lib

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

lib: $(OBJS)
	$(NVCC) -lib -o libgpu_symplectic.a $(OBJS)

test: lib
	$(NVCC) $(NVCC_FLAGS) tests/test_correctness.cu -L. -lgpu_symplectic $(LIBS) -o test_correctness
	./test_correctness

bench: lib
	$(NVCC) $(NVCC_FLAGS) bench/benchmark.cu -L. -lgpu_symplectic $(LIBS) -o benchmark
	./benchmark

clean:
	rm -rf $(OBJDIR) libgpu_symplectic.a test_correctness benchmark
