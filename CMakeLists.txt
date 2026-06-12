cmake_minimum_required(VERSION 3.22)

project(cuda_neural_net_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)

add_library(cpu_reference STATIC
    src/cpu_reference.cpp
)
target_include_directories(cpu_reference PUBLIC include)

add_executable(cpu_reference_test
    tests/test_cpu_reference.cpp
)
target_link_libraries(cpu_reference_test PRIVATE cpu_reference)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(cuda_inference STATIC
        src/activation.cu
        src/inference_engine.cu
        src/matmul.cu
        src/quantize.cu
    )
    target_include_directories(cuda_inference PUBLIC include)
    target_link_libraries(cuda_inference PUBLIC cpu_reference)
    set_target_properties(cuda_inference PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86"
    )

    add_executable(inference src/main.cu)
    target_link_libraries(inference PRIVATE cuda_inference)
    set_target_properties(inference PROPERTIES CUDA_ARCHITECTURES "75;80;86")

    add_executable(gpu_kernel_test tests/test_gpu_kernels.cu)
    target_link_libraries(gpu_kernel_test PRIVATE cuda_inference)
    set_target_properties(gpu_kernel_test PROPERTIES CUDA_ARCHITECTURES "75;80;86")
else()
    message(STATUS "CUDA compiler not found. Building CPU reference test only.")
endif()
