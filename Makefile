NVCC = nvcc
ARCH = -arch=sm_89
FLAGS = -O3 $(ARCH) -Iinclude
SRCS = main.cu src/kernels.cu src/benchmark.cu src/test_correctness.cu src/test_weight_loader.cu
TARGET = engine_p1

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(FLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean