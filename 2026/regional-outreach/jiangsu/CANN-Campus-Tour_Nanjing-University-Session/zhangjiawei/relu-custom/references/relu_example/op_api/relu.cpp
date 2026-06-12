// relu.cpp — L0 API 实现
#include "relu.h"
#include "op_common/op_api/util/op_api.h"

aclnnStatus aclnnReluGetWorkspaceSize(
    const aclTensor* xDesc, const aclTensor* yDesc,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    aclnnStatus ret = aclSetTensorShape(yDesc, xDesc->shape.dimNum, xDesc->shape.dims);
    if (ret != ACLNN_SUCCESS) return ret;
    ret = aclnnCreateExecutor("Relu", xDesc, yDesc, executor);
    if (ret != ACLNN_SUCCESS) return ret;
    *workspaceSize = 0;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRelu(void* workspace, uint64_t workspaceSize,
                       aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnTensorAicoreExecute(nullptr, nullptr, executor, stream);
}
