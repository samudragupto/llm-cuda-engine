@echo off
nvcc -O3 -arch=sm_89 -Iinclude main.cu src/kernels.cu src/benchmark.cu src/test_correctness.cu src/test_weight_loader.cu -o engine_p1.exe
if %errorlevel% equ 0 (
    echo Build successful!
    engine_p1.exe %*
) else (
    echo Build failed!
)