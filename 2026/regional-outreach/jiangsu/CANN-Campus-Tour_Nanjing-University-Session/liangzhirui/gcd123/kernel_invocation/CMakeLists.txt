cmake_minimum_required(VERSION 3.16)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 17)
project(gcd123_kernel_invocation)

# ---------------- NPU 模式（ccec 编译 kernel + host 主程序） ----------------
function(add_ccec_target)
    cmake_parse_arguments(NPU "" "TARGET;PRODUCT;CORE;OUTPUT_FILE" "CCE_FILE;CPP_FILE" ${ARGN})
    set(custom_inc_opts
        "-I${install_path}/acllib/include"
        "-I${install_path}/compiler/tikcpp/tikcfw"
        "-I${install_path}/compiler/tikcpp/tikcfw/impl"
        "-I${install_path}/compiler/tikcpp/tikcfw/interface"
        "-I${install_path}/tools/tikicpulib/lib/include"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/.."
        "-DGCD123_DIRECT_INVOCATION"
    )
    set(ccec_base_opts
        "-O2" "-std=c++17"
        "-L${install_path}/atc/lib64"
        "-lruntime" "-lascendcl" "-lstdc++"
    )
    set(ccec_extopts_ascend910_AiCore
        "--cce-aicore-arch=dav-c100"
    )
    set(ccec_extopts_ascend910
        "-L${install_path}/toolkit/tools/simulator/Ascend910A/lib"
        "-L${install_path}/tools/tikicpulib/lib/Ascend910A"
    )
    set(ccec_extopts_ascend310p_AiCore
        "--cce-aicore-arch=dav-m200"
    )
    set(ccec_extopts_ascend310p_VectorCore
        "--cce-aicore-arch=dav-m200-vec"
    )
    set(ccec_extopts_ascend310p
        "-mllvm" "-cce-aicore-function-stack-size=16000"
        "-mllvm" "-cce-aicore-fp-ceiling=2"
        "-mllvm" "-cce-aicore-record-overflow=false"
        "-L${install_path}/toolkit/tools/simulator/Ascend310P1/lib"
        "-L${install_path}/tools/tikicpulib/lib/Ascend310P1"
    )
    set(ccec_extopts_ascend910b_AiCore
        "--cce-aicore-arch=dav-c220-vec"
    )
    set(ccec_extopts_ascend910b
        "-L${install_path}/toolkit/tools/simulator/Ascend910B1/lib"
        "-L${install_path}/tools/tikicpulib/lib/Ascend910B1"
    )

    add_custom_target(${NPU_TARGET} ALL
        COMMAND
            ccec -x cce ${NPU_CCE_FILE} -x c++ ${NPU_CPP_FILE}
            ${ccec_extopts_${NPU_PRODUCT}_${NPU_CORE}}
            ${ccec_extopts_${NPU_PRODUCT}}
            ${custom_inc_opts} ${ccec_base_opts}
            "-o" ${NPU_OUTPUT_FILE}
    )
endfunction()

set(cce_src
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/gcd123.cpp
)
set(cpp_src
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_ccec_target(TARGET gcd123_npu
    PRODUCT ${product_type}
    CORE ${core_type}
    CCE_FILE ${cce_src}
    CPP_FILE ${cpp_src}
    OUTPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/gcd123_npu
)

# ---------------- CPU 仿真调试模式（tikicpulib，无需 NPU 硬件） ----------------
if (NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${install_path}/tools/tikicpulib/lib/cmake)
endif()
find_package(tikicpulib REQUIRED)

add_executable(gcd123_cpu
    ${cpp_src}
    ${cce_src}
)

target_include_directories(gcd123_cpu PRIVATE
    ${install_path}/acllib/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(gcd123_cpu PRIVATE
    GCD123_DIRECT_INVOCATION
)

target_link_libraries(gcd123_cpu PRIVATE
    tikicpulib::${product_type}
    ascendcl
)

target_compile_options(gcd123_cpu PRIVATE
    -g
)

set_target_properties(gcd123_cpu PROPERTIES
    OUTPUT_NAME gcd123_cpu
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
