if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tf_plugin")
    add_subdirectory(tf_plugin)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_plugin")
    add_subdirectory(onnx_plugin)
endif()
