cmake_minimum_required(VERSION 3.24)
project(dettato_ggml VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(DETTATO_BUILD_TOOLS "Build the CLI, parity and benchmark tools" ${PROJECT_IS_TOP_LEVEL})
option(DETTATO_BUILD_TESTS "Build the test suite (fetches Catch2)" ${PROJECT_IS_TOP_LEVEL})
option(DETTATO_NATIVE "Optimize ggml for the build machine's CPU (never for a shipped build)" OFF)

# --- ggml -----------------------------------------------------------------
#
# A host that already builds ggml (NeuralNote builds it for muscriptor.cpp) provides the targets, and this library links
# against them: a binary can only hold one ggml. The host's ggml can be v1.0.0 or the v0.23 fork muscriptor.cpp pins
# (DamRsn/ggml b01bbb8; src/ggml_compat.hpp). Otherwise ggml v1.0.0 is fetched; to use a local checkout, configure with
#   -DFETCHCONTENT_SOURCE_DIR_GGML=/path/to/ggml
set(DETTATO_GGML_COMMIT 417a024b4fb6bc3116d41af7c067166a039bdf12)   # v1.0.0

if(NOT TARGET ggml)
    include(FetchContent)
    set(BUILD_SHARED_LIBS OFF)
    set(GGML_STATIC       ON  CACHE BOOL "" FORCE)
    set(GGML_OPENMP       OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
    set(GGML_ACCELERATE   OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE ${DETTATO_NATIVE} CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE    ON  CACHE BOOL "" FORCE) # 2-4x faster small-K f32 matmuls on the CPU (drum and grouper graphs)

    if(APPLE)
        set(DETTATO_METAL_DEFAULT ON)
        set(DETTATO_VULKAN_DEFAULT OFF)
    else()
        set(DETTATO_METAL_DEFAULT OFF)
        find_package(Vulkan QUIET COMPONENTS glslc)
        set(DETTATO_VULKAN_DEFAULT ${Vulkan_FOUND})
    endif()
    option(DETTATO_METAL  "Build the ggml Metal backend"  ${DETTATO_METAL_DEFAULT})
    option(DETTATO_VULKAN "Build the ggml Vulkan backend" ${DETTATO_VULKAN_DEFAULT})
    set(GGML_METAL  ${DETTATO_METAL}  CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${DETTATO_VULKAN} CACHE BOOL "" FORCE)
    if(DETTATO_METAL)
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        ${DETTATO_GGML_COMMIT})
    FetchContent_MakeAvailable(ggml)
endif()

# --- the library ----------------------------------------------------------
# Every .cpp under src/ (stages add files without touching this list).
file(GLOB_RECURSE DETTATO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(dettato_ggml STATIC ${DETTATO_SOURCES})
target_include_directories(dettato_ggml
    PUBLIC  include
    PRIVATE src third_party/pocketfft)
target_link_libraries(dettato_ggml PUBLIC ggml)
target_compile_features(dettato_ggml PUBLIC cxx_std_23)
if(MSVC)
    target_compile_options(dettato_ggml PRIVATE /W4 /utf-8 /bigobj)
    target_compile_definitions(dettato_ggml PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(dettato_ggml PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()
# No floating-point contraction (GCC's default in ISO mode, Clang's is "on"): the host code reproduces numpy / scipy /
# torch arithmetic, and every fused multiply-add it wants is an explicit std::fma.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(dettato_ggml PRIVATE -ffp-contract=off)
endif()

if(DETTATO_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(DETTATO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
