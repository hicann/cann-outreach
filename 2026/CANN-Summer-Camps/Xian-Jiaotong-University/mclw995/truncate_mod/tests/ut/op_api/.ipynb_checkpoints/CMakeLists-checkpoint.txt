set(OP_API_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_api)

add_executable(truncate_mod_op_api_ut
    test_truncate_mod_api.cpp
    ${OP_API_SRC_DIR}/aclnn_truncate_mod.cpp
)
target_compile_definitions(truncate_mod_op_api_ut PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(truncate_mod_op_api_ut PRIVATE
    ${OP_API_SRC_DIR}
    ${UT_COMMON_INCLUDE_DIRS}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include/aclnn
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/aclnn
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
)

target_link_directories(truncate_mod_op_api_ut PRIVATE
    ${UT_COMMON_LIB_DIRS}
)

target_link_libraries(truncate_mod_op_api_ut PRIVATE
    gtest
    gtest_main
    acl_op_compiler
    ascendcl
    nnopbase
)

add_dependencies(truncate_mod_op_api_ut third_party_gtest)

set_target_properties(truncate_mod_op_api_ut PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
