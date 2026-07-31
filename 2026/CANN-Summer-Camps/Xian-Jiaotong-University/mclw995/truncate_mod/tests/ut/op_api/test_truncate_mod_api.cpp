#include <gtest/gtest.h>
#include "aclnn_truncate_mod.h"

TEST(TruncateModApiTest, ApiSymbolExists)
{
    auto getWorkspaceFunc = &aclnnTruncateModGetWorkspaceSize;
    auto executeFunc = &aclnnTruncateMod;

    EXPECT_NE(getWorkspaceFunc, nullptr);
    EXPECT_NE(executeFunc, nullptr);
}
