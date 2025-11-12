cmake_minimum_required(VERSION 3.10)
project(gridCuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# CUDA compute capability
# You may need to adjust these based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 86)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Define the CUDA source file
set(CUDA_SOURCES src/python_bindings.cu)

# Set CUDA-specific properties
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Add pybind11 module with CUDA source
pybind11_add_module(gridCuda ${CUDA_SOURCES})

# Make sure the extension is built as a shared library
set_target_properties(gridCuda PROPERTIES PREFIX "")

# Link CUDA libraries
target_link_libraries(gridCuda PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Add CUDA compilation flags
target_compile_options(gridCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --diag-suppress 20042>)

# Specify that this is a CUDA compilation
set_target_properties(gridCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Define the output directory for the compiled Python module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Set the output directory for the Python module
set_target_properties(gridCuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gridCuda
)

# Add installation rules
install(TARGETS gridCuda DESTINATION ${Python_SITEARCH})
