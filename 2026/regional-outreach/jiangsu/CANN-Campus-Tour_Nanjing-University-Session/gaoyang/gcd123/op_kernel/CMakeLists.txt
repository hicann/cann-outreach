# Kernel-side build: cross-compiles gcd123.cpp into the AICore binary.
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910b4" CACHE STRING "Target SoC version")
endif()

# ARCH / os / compiler are normally provided by the CANN toolchain preset;
# fall back to sane defaults for local builds.
if(NOT DEFINED ARCH)
    set(ARCH "aarch64")
endif()
if(NOT DEFINED os)
    set(os "linux")
endif()

file(GLOB KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# ascending build of the kernel binary (kbuild-based, official flow)
add_library(gcd123_kernel_obj OBJECT ${KERNEL_SRC})
target_include_directories(gcd123_kernel_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_host
)
target_compile_options(gcd123_kernel_obj PRIVATE
        -fno-c++-static-destructors
        -mllvm -licm-stability-only
        -Wno-deprecated-declarations
)
