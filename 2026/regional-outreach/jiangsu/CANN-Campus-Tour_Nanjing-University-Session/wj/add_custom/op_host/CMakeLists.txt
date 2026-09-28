# ----------------------------------------------------------------------------
# add_custom 算子 host 侧 CMakeLists
# add_modules_sources 会收集本目录源码并构建为 host 插件，
# OPTYPE 指定算子名（小写下划线形式），ACLNNTYPE 指定 aclnn 接口类型
# ----------------------------------------------------------------------------

add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE add_custom ACLNNTYPE aclnn)
