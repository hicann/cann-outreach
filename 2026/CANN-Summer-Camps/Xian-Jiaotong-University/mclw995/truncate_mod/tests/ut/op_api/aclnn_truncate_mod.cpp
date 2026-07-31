#include <string.h>
#include "graph/types.h"
#include "aclnn_truncate_mod.h"

#ifdef ACLNN_WITH_BINARY
#include <vector>
#include <tuple>
#include <map>
#include "graph/ascend_string.h"
#include "TruncateMod_op_resource.h"
using OP_HOST_FUNC_HANDLE = std::vector<void *>;
using OP_RES = std::tuple<const uint8_t *, const uint8_t *>;
using OP_BINARY_RES = std::vector<OP_RES>;
using OP_RUNTIME_KB_RES = std::vector<OP_RES>;
using OP_RESOURCES = std::map<ge::AscendString,
    std::tuple<OP_HOST_FUNC_HANDLE, OP_BINARY_RES, OP_RUNTIME_KB_RES>>;
using OP_SOC_RESOURCES = std::map<ge::AscendString, std::tuple<OP_HOST_FUNC_HANDLE,
    std::map<ge::AscendString, OP_BINARY_RES>, OP_RUNTIME_KB_RES>>;
namespace op {
extern uint32_t GenOpTypeId(const char *op_name, const OP_RESOURCES &op_resources);
extern uint32_t GenOpTypeId(const char *op_name, const OP_SOC_RESOURCES &op_resources);
}
#endif

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B,
    SOC_VERSION_ASCEND910_93,
    SOC_VERSION_ASCEND910_95,
    SOC_VERSION_ASCEND310P,
    SOC_VERSION_ASCEND310B,
    SOC_VERSION_BS9SX1A,
    SOC_VERSION_ASCEND610Lite,
    SOC_VERSION_ASCEND910_55,
    SOC_VERSION_MC61AM21A,
    SOC_VERSION_MC62CM12A,
    SOC_VERSION_BS9SX2A,
    SOC_VERSION_ASCEND910_96,
    SOC_VERSION_KIRINX90,
    SOC_VERSION_KIRIN9030
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910B};
uint32_t socSupportListLen = 1;

TensorDesc inputDesc0_0[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[2] =
    {{ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc inputDesc0_3[2] =
    {{ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc inputDesc0_4[2] =
    {{ge::DT_INT8, ge::FORMAT_ND},
     {ge::DT_INT8, ge::FORMAT_ND}};
TensorDesc inputDesc0_5[2] =
    {{ge::DT_UINT8, ge::FORMAT_ND},
     {ge::DT_UINT8, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[1] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[1] =
    {{ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[1] =
    {{ge::DT_BF16, ge::FORMAT_ND}};
TensorDesc outputDesc0_3[1] =
    {{ge::DT_INT32, ge::FORMAT_ND}};
TensorDesc outputDesc0_4[1] =
    {{ge::DT_INT8, ge::FORMAT_ND}};
TensorDesc outputDesc0_5[1] =
    {{ge::DT_UINT8, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 2, outputDesc0_0, 1};
SupportInfo list0_1 = {inputDesc0_1, 2, outputDesc0_1, 1};
SupportInfo list0_2 = {inputDesc0_2, 2, outputDesc0_2, 1};
SupportInfo list0_3 = {inputDesc0_3, 2, outputDesc0_3, 1};
SupportInfo list0_4 = {inputDesc0_4, 2, outputDesc0_4, 1};
SupportInfo list0_5 = {inputDesc0_5, 2, outputDesc0_5, 1};
SupportInfo supportInfo0[6] = {list0_0, list0_1, list0_2, list0_3, list0_4, list0_5};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 6};

OpSocSupportInfo opSocSupportList[1] = {socSupportInfo0};
OpSupportList supportList = {opSocSupportList, 1};

[[maybe_unused]] uint32_t NNOPBASE_TruncateMod = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIgnoreContinuesInput(void *executor,
                                                   const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIntArrayInput(void *executor, const aclIntArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddBoolArrayInput(void *executor, const aclBoolArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddFloatArrayInput(void *executor, const aclFloatArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicInput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicOutput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index, const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray* array, const size_t index);
extern aclnnStatus NnopbaseAddFloatArrayAttr(void *executor, const aclFloatArray* array, const size_t index);
extern aclnnStatus NnopbaseAddBoolArrayAttr(void *executor, const aclBoolArray* array, const size_t index);
extern aclnnStatus NnopbaseAddArrayAttrWithDtype(void *executor, void *array, const size_t len, const size_t elementSize, const size_t index, const NnopbaseAttrDtype dtype);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace, uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);
extern aclnnStatus NnopbaseAddScalarInput(void *executor, const aclScalar *scalar, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern aclnnStatus NnopbaseAddScalarListInput(void *executor, const aclScalarList *scalarList, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern void NnopbaseAddOpTypeId(void *executor, const uint32_t opTypeId);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name, const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern aclnnStatus NnopbaseSetRef(void *executor, const size_t inputIrIdx, const size_t outputIrIdx);
extern void __attribute__((weak)) NnopbaseSetMatchArgsFlag(void *executor);
extern bool __attribute__((weak)) NnopbaseMatchArgs(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseGetUnContiguousTensors(void *executor, const aclTensorList **inTensors);
extern aclnnStatus NnopbaseSetUnContExecutor(void *executor, aclOpExecutor *inExe, const size_t inWsSize);
extern aclnnStatus NnopbaseGetUnContExecutor(void *executor, aclOpExecutor **inExe, size_t *inWsSize);
extern aclnnStatus NnopbaseGetRefUnContiguousTensors(void *executor, const aclTensorList **unContTensors, const aclTensorList **contTensors);
extern aclnnStatus NnopbaseSetViewCopyExecutor(void *executor, aclOpExecutor *exe);
extern aclnnStatus NnopbaseGetViewCopyExecutor(void *executor, aclOpExecutor **exe);
extern aclnnStatus NnopbaseReleaseRefContiguousTensors(void *executor, const aclTensorList **tensors);
extern void *NnopbaseGetApiFunc(const char *funcName);
using AclnnContiguousGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnViewCopyGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

#define ACLNN_SUCCESS  0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

aclnnStatus aclnnTruncateModGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
#ifdef ACLNN_WITH_BINARY
    static uint32_t TruncateModOpTypeId = op::GenOpTypeId("TruncateMod", TruncateMod_RESOURCES);
#endif
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnTruncateModTiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "TruncateMod";
    char inputDesc[] = {1, 1};
    char outputDesc[] = {1};
    char attrDesc[] = {};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(x1);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(x2);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(out);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    if (NnopbaseSetMatchArgsFlag != NULL) {
        NnopbaseSetMatchArgsFlag(*executor);
    }
#ifdef ACLNN_WITH_BINARY
    NnopbaseAddOpTypeId(*executor, TruncateModOpTypeId);
#endif
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x1, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, x2, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, out, 0));
    if (NnopbaseMatchArgs != NULL) {
        if (NnopbaseMatchArgs(*executor, workspaceSize)) {
            NnopbaseReportApiInfo(timeStamp, dfxId);
            return ACLNN_SUCCESS;
        }
    }
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "x1", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "x2", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "out", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));

    const aclTensorList *inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(*executor, &inUnContTensors);
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    if (inUnContTensors != nullptr) {
        static AclnnContiguousGetWorkspaceSizeFunc aclnnContiguousGetWorkspaceSize = (AclnnContiguousGetWorkspaceSizeFunc)NnopbaseGetApiFunc("aclnnContiguousGetWorkspaceSize");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguousGetWorkspaceSize);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor));
    }
    NnopbaseSetUnContExecutor(*executor, aclInExecutor, inContWorkspaceSize);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    *workspaceSize += inContWorkspaceSize;
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTruncateMod(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    NnopbaseGetUnContExecutor(executor, &aclInExecutor, &inContWorkspaceSize);
    if (workspaceSize < inContWorkspaceSize) {
        NnopbaseOpLogE(ACLNN_ERR_PARAM_INVALID, "input workspaceSize must be larger than contiguous size!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    workspaceSize -= inContWorkspaceSize;
    void *inWorkspace = (char *)workspace + workspaceSize;
    if (aclInExecutor != nullptr) {
        static AclnnFunc aclnnContiguous = (AclnnFunc)NnopbaseGetApiFunc("aclnnContiguous");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguous);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
