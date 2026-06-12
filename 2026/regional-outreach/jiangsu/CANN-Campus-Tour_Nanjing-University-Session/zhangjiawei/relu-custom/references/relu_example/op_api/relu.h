// relu.h — L0 API 声明
#ifndef ASCENDC_RELU_H
#define ASCENDC_RELU_H
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnReluGetWorkspaceSize(const aclTensor* xDesc, const aclTensor* yDesc,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnRelu(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
