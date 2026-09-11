# 按实际存在的目录自动适配 host 侧路径：aarch64-linux / x86_64-linux（8.x 及部分 9.x 布局）/ host-libs（9.x 新布局）
if(EXISTS $ENV{ASCEND_HOME_PATH}/aarch64-linux)
    set(OP_HOST_LIB_PATH $ENV{ASCEND_HOME_PATH}/aarch64-linux)
elseif(EXISTS $ENV{ASCEND_HOME_PATH}/x86_64-linux)
    set(OP_HOST_LIB_PATH $ENV{ASCEND_HOME_PATH}/x86_64-linux)
else()
    set(OP_HOST_LIB_PATH $ENV{ASCEND_HOME_PATH}/host-libs)
endif()

file(GLOB host_ops_def_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/*def.cpp
)

file(GLOB host_ops_infershape_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/*_infershape.cpp
)

set(host_ops_tiling_srcs)
file(GLOB TILING_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*tiling.cpp)
list(APPEND host_ops_tiling_srcs ${TILING_FILES})

set(host_ops_srcs
    ${host_ops_def_srcs}
    ${host_ops_infershape_srcs}
    ${host_ops_tiling_srcs}
)

npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    COMPILE_OPTIONS
        -I${OP_HOST_LIB_PATH}/include
        -I${OP_HOST_LIB_PATH}/asc/include/tiling
        -I${OP_HOST_LIB_PATH}/pkg_inc
        -I${OP_HOST_LIB_PATH}/pkg_inc/op_common
        -I${OP_HOST_LIB_PATH}/pkg_inc/base
        -I${OP_HOST_LIB_PATH}/pkg_inc/exe_graph
        -I${OP_HOST_LIB_PATH}/pkg_inc/graph
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

target_include_directories(cust_optiling PRIVATE
    ${OP_HOST_LIB_PATH}/include
    ${OP_HOST_LIB_PATH}/asc/include/tiling
    ${OP_HOST_LIB_PATH}/pkg_inc
    ${OP_HOST_LIB_PATH}/pkg_inc/op_common
    ${OP_HOST_LIB_PATH}/pkg_inc/base
)

set(op_api_dir ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
if(EXISTS ${op_api_dir} AND IS_DIRECTORY ${op_api_dir})
    file(GLOB op_api_srcs ${op_api_dir}/*.cpp)
else()
    file(GLOB op_api_srcs "${CMAKE_BINARY_DIR}/autogen/aclnn_*.cpp")
endif()

npu_op_library(cust_opapi ACLNN
    ${op_api_srcs}
)

target_include_directories(cust_opapi PRIVATE
    ${op_api_dir}
    ${OP_HOST_LIB_PATH}/include
    ${OP_HOST_LIB_PATH}/include/aclnn
    ${OP_HOST_LIB_PATH}/asc/include
    ${OP_HOST_LIB_PATH}/pkg_inc
    ${OP_HOST_LIB_PATH}/pkg_inc/op_common
    ${OP_HOST_LIB_PATH}/pkg_inc/base
    ${OP_HOST_LIB_PATH}/pkg_inc/aicpu
)

target_compile_options(cust_opapi PRIVATE -UACLNN_WITH_BINARY)

file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${proto_src} PROPERTIES GENERATED TRUE)

npu_op_library(cust_op_proto GRAPH
    ${host_ops_srcs}
    ${proto_src}
)

target_include_directories(cust_op_proto PRIVATE
    ${OP_HOST_LIB_PATH}/include
    ${OP_HOST_LIB_PATH}/asc/include/tiling
    ${OP_HOST_LIB_PATH}/pkg_inc
    ${OP_HOST_LIB_PATH}/pkg_inc/op_common
    ${OP_HOST_LIB_PATH}/pkg_inc/base
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_optiling
        cust_op_proto
        cust_opapi
)

# Install hand-written op_api headers into RUN package
if(EXISTS ${op_api_dir} AND IS_DIRECTORY ${op_api_dir})
    file(GLOB op_api_aclnn_headers ${op_api_dir}/aclnn_*.h ${op_api_dir}/acl_*.h)
    if(op_api_aclnn_headers)
        npu_op_package_add(${package_name}
            FILES ${op_api_aclnn_headers}
            TYPE ACLNN
        )
    endif()
endif()
