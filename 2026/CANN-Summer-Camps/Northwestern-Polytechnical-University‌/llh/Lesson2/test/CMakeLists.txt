cmake_minimum_required(VERSION 3.16)
project(test_sub_custom_template)

find_package(ASC REQUIRED)

add_executable(execute_op main.cpp)
target_link_libraries(execute_op
    cust_opapi
    nnopbase
    acl_rt
    pthread
    dl
)
