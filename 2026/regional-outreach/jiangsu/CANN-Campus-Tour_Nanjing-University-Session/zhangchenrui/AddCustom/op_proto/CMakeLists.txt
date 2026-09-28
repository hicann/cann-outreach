file(GLOB_RECURSE PROTO_SRCS *.cpp)
add_library(add_custom_proto SHARED ${PROTO_SRCS})
target_include_directories(add_custom_proto PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(add_custom_proto PRIVATE ascendc_proto)