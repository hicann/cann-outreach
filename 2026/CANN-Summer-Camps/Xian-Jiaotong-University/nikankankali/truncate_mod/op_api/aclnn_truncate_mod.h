#ifndef ACLNN_TRUNCATE_MOD_H_
#define ACLNN_TRUNCATE_MOD_H_
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnTruncateMod(const aclTensor* x1, const aclTensor* x2,
                              const aclTensor* y, aclOpExecutor* executor);
#ifdef __cplusplus
}
#endif
#endif
