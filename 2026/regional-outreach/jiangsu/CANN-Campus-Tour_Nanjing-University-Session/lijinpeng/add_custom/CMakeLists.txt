# ---------------------------------------------------------------------------
# add_custom 算子构建脚本（Kernel Launch 直调方式）
#
# 构建产物：
#   ${CMAKE_BINARY_DIR}/add_custom        Host 侧可执行程序
#   以及由 ascendc_library 生成的 aclrtlaunch_add_custom.h 和算子二进制
#
# 用法：
#   mkdir -p build && cd build
#   cmake ..
#   make -j
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(add_custom CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ---------------------------------------------------------------------------
# 1. 定位 CANN 安装路径（ASCEND_HOME_PATH）
# ---------------------------------------------------------------------------
if(NOT ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        foreach(candidate
                "/usr/local/Ascend/ascend-toolkit/latest"
                "/usr/local/Ascend/latest"
                "C:/Program Files/Ascend/ascend-toolkit/latest")
            if(EXISTS "${candidate}/compiler/tikcpp")
                set(ASCEND_HOME_PATH "${candidate}")
                break()
            endif()
        endforeach()
    endif()
endif()

if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR
        "未找到 CANN 安装路径。请先 source 环境变量脚本（例如 "
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh）或用 -DASCEND_HOME_PATH=<path> 指定。")
endif()
message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")

# ---------------------------------------------------------------------------
# 2. 定位官方 ascendc.cmake（不同 CANN 版本路径略有差异，按候选路径查找）
# ---------------------------------------------------------------------------
set(ASCENDC_CMAKE_FILE "")
foreach(candidate
        "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
        "${ASCEND_HOME_PATH}/compiler/ascendc_kernel_cmake"
        "${ASCEND_HOME_PATH}/tools/ascendc_kernel_cmake"
        "${ASCEND_HOME_PATH}/ascendc_kernel_cmake")
    if(EXISTS "${candidate}/ascendc.cmake")
        set(ASCENDC_CMAKE_FILE "${candidate}/ascendc.cmake")
        break()
    endif()
endforeach()

if(NOT ASCENDC_CMAKE_FILE)
    # 兜底：在有限深度内搜索
    file(GLOB ASCENDC_CMAKE_CANDIDATES
         "${ASCEND_HOME_PATH}/*/ascendc.cmake"
         "${ASCEND_HOME_PATH}/*/*/ascendc.cmake"
         "${ASCEND_HOME_PATH}/*/*/*/ascendc.cmake"
         "${ASCEND_HOME_PATH}/*/*/*/*/ascendc.cmake")
    list(LENGTH ASCENDC_CMAKE_CANDIDATES candidateCount)
    if(candidateCount GREATER 0)
        list(GET ASCENDC_CMAKE_CANDIDATES 0 ASCENDC_CMAKE_FILE)
    endif()
endif()

if(NOT ASCENDC_CMAKE_FILE)
    message(FATAL_ERROR "未找到 ascendc.cmake，请确认 CANN 安装是否完整。")
endif()
message(STATUS "ascendc.cmake = ${ASCENDC_CMAKE_FILE}")

include("${ASCENDC_CMAKE_FILE}")

# ---------------------------------------------------------------------------
# 3. 编译算子 Kernel（生成 aclrtlaunch_add_custom.h）
#
# 注意：不同 CANN 版本的示例工程中 ascendc_library 使用 STATIC 或 SHARED，
#      若链接阶段报未定义符号（aclrtlaunch_add_custom 相关），
#      可用 -DADD_CUSTOM_KERNEL_LIB_TYPE=STATIC 切换后重新构建。
# ---------------------------------------------------------------------------
set(ADD_CUSTOM_KERNEL_LIB_TYPE SHARED CACHE STRING "ascendc_library 的库类型（STATIC / SHARED）")
ascendc_library(add_custom_kernel ${ADD_CUSTOM_KERNEL_LIB_TYPE} add_custom.cpp)

# ascendc_include_directories 为官方提供的封装宏，版本较老时可能不存在
if(COMMAND ascendc_include_directories)
    ascendc_include_directories(add_custom_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
else()
    message(STATUS "ascendc_include_directories 不可用，改用普通 include 目录设置")
    target_include_directories(add_custom_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# 4. 编译 Host 侧可执行程序
# ---------------------------------------------------------------------------
add_executable(add_custom main.cpp)
target_include_directories(add_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/runtime/include
    ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw)
target_link_libraries(add_custom PRIVATE add_custom_kernel)

# 链接 AscendCL 运行时（不同版本库路径不同，按候选路径查找）
find_library(ASCENDCL_LIB ascendcl
    PATHS
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/runtime/lib64
        ${ASCEND_HOME_PATH}/lib
        ${ASCEND_HOME_PATH}/lib64/stub
        ${ASCEND_HOME_PATH}/runtime/lib64/stub
    NO_DEFAULT_PATH)
if(ASCENDCL_LIB)
    message(STATUS "ascendcl = ${ASCENDCL_LIB}")
    target_link_libraries(add_custom PRIVATE ${ASCENDCL_LIB})
else()
    message(WARNING "未找到 libascendcl，将依赖 ascendc_library 的默认链接设置。")
endif()
