# Host 侧编译为普通动态库
file(GLOB_RECURSE HOST_SRCS *.cpp)
add_library(add_custom_host SHARED ${HOST_SRCS})
target_include_directories(add_custom_host PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(add_custom_host PRIVATE ascendc_host)