@echo off
nvcc -O3 -arch=sm_89 -Iinclude main.cu src/kernels.cu src/benchmark.cu src/test_correctness.cu src/test_weight_loader.cu src/phase2.cu src/phase3.cu -o engine_p3.exe
if %errorlevel% neq 0 (
    echo Build failed
    exit /b 1
)
engine_p3.exe %*