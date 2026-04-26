cmake_minimum_required(VERSION 3.12)
project(turboquant_pybind)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(pybind11
    URL https://github.com/pybind/pybind11/archive/v2.6.2.tar.gz
)
FetchContent_MakeAvailable(pybind11)

# Create the pybind11 module
pybind11_add_module(turboquant_pybind
    turboQuant/turboquant_pybind.cpp
)

# Link with Python
target_link_libraries(turboquant_pybind PRIVATE Python3::Python)

# Optimization flags
if(MSVC)
    target_compile_options(turboquant_pybind PRIVATE /O2)
else()
    target_compile_options(turboquant_pybind PRIVATE -O3 -march=native)
endif()

# Installation
install(TARGETS turboquant_pybind LIBRARY DESTINATION .)
