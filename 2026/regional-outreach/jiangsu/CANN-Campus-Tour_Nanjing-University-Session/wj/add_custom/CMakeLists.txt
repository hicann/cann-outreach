# ----------------------------------------------------------------------------
# add_custom 算子 CMake 入口
# 结构对齐 CANN 算子仓库（如 cann/ops-nn examples）：
#   op_host/  构建 host 侧插件（def / infershape / tiling）
#   op_graph/ 构建 graph 插件（proto 注册）
#   examples/ aclnn 调用示例（需要 ENABLE_TEST/BENCHMARK 才构建）
# ----------------------------------------------------------------------------

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST AND NOT BENCHMARK)
    list(REMOVE_ITEM CURRENT_DIRS tests)
    list(REMOVE_ITEM CURRENT_DIRS examples)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
