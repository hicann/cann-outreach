/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <map>
#include <cstdint>
#include "acl/acl.h"
#include "../common/sampleDevice.h"
#include "../common/sampleModel.h"
#include "../common/utils.h"
using namespace std;

bool g_isDevice = false;

class SampleRes50ImagenetClassification {
 public:
  SampleRes50ImagenetClassification() = default;
  ~SampleRes50ImagenetClassification() = default;
  Result InitResource(const char *modelPath);
  Result Process();
  Result PrepareModel(const char *modelPath);
  void OutputModelResult();
 private:
  // Device resource.
  std::shared_ptr<AclInstance> aclInstance_;
  std::shared_ptr<AclDevice> device_;
  std::shared_ptr<AclContext> context_;
  std::shared_ptr<AclStream> stream_;
  // Model.
  std::shared_ptr<AclModelWeight> modelWeight_;
  std::shared_ptr<AclModelWork> modelWork_;
  std::shared_ptr<AclModelDesc> modelDesc_;
  std::shared_ptr<AclModelInput> modelInput_;
  std::shared_ptr<AclModelOutput> modelOutput_;
  uint32_t modelId_;
};

Result SampleRes50ImagenetClassification::InitResource(const char *modelPath) {
  // ACL init.
  aclInstance_ = std::make_shared<AclInstance>(modelPath);
  INFO_LOG("acl init success");
  // Set device.
  device_ = std::make_shared<AclDevice>(0);
  INFO_LOG("set device success");
  // Create context (set current).
  context_ = std::make_shared<AclContext>(0);
  INFO_LOG("create context success");
  // Create stream.
  stream_ = std::make_shared<AclStream>();
  INFO_LOG("create stream success");
  // Get run mode
  // runMode is ACL_HOST which represents app is running in host
  // runMode is ACL_DEVICE which represents app is running in device
  aclrtRunMode runMode;
  auto ret = aclrtGetRunMode(&runMode);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  g_isDevice = (runMode == ACL_DEVICE);
  return SUCCESS;
}

Result SampleRes50ImagenetClassification::PrepareModel(const char *modelPath) {
  // Load model.
  size_t modelWorkSize, modelWeightSize;
  aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize, &modelWeightSize);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  modelWork_ = std::make_shared<AclModelWork>(modelWorkSize);
  modelWeight_ = std::make_shared<AclModelWeight>(modelWeightSize);
  ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWork_->GetModelWorkPtr(), modelWork_->GetModelWorkSize(),
                                  modelWeight_->GetModelWeightPrt(), modelWeight_->GetModelWeightSize());
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  INFO_LOG("load model %s success.", modelPath);
  // Create ModelDesc.
  modelDesc_ = std::make_shared<AclModelDesc>(modelId_);
  return SUCCESS;
}

// 【核心修改：完全按照你提供的aclFloat16规范改写OutputModelResult】
void SampleRes50ImagenetClassification::OutputModelResult() {
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(modelOutput_->GetDataSet()); ++i) {
    // 1、获取输出数据缓冲区地址（不变）
    aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(modelOutput_->GetDataSet(), i);
    void* outputBuffer = aclGetDataBufferAddr(dataBuffer);
    uint32_t bufByteLen = aclGetDataBufferSizeV2(dataBuffer);

    void* hostCopyBuf = nullptr;
    aclFloat16* outputData = nullptr;

    if (!g_isDevice) {
      // Host场景：分配主机内存，Device -> Host拷贝输出数据
      aclError ret = aclrtMallocHost(&hostCopyBuf, bufByteLen);
      if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtMallocHost failed, len[%u], err[%d]", bufByteLen, static_cast<int32_t>(ret));
        return;
      }
      ret = aclrtMemcpy(hostCopyBuf, bufByteLen, outputBuffer, bufByteLen, ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy dev2host fail err[%d]", static_cast<int32_t>(ret));
        (void)aclrtFreeHost(hostCopyBuf);
        return;
      }
      // 关键修改：缓冲区转 aclFloat16*
      outputData = static_cast<aclFloat16*>(hostCopyBuf);
    } else {
      // Device直跑场景：直接强转设备地址为aclFloat16*
      outputData = static_cast<aclFloat16*>(outputBuffer);
    }

    // 计算FP16元素个数：总字节 / 单个aclFloat16占2字节
    size_t elemNum = bufByteLen / sizeof(aclFloat16);
    // map：key=浮点置信度(降序)，value=分类索引
    map<float, unsigned int, greater<float>> resultMap;

    for (unsigned int j = 0; j < elemNum; j++) {
      // 2、aclFloat16 转 float，使用CANN官方接口
      float confFloat = aclFloat16ToFloat(outputData[j]);
      resultMap[confFloat] = j;
    }

    // 打印Top5结果
    int cnt = 0;
    for (auto& item : resultMap) {
      if (++cnt > 5) break;
      INFO_LOG("top %d: index[%u] confidence[%lf]", cnt, item.second, item.first);
    }

    // 释放Host临时内存
    if (!g_isDevice && hostCopyBuf != nullptr) {
      aclError ret = aclrtFreeHost(hostCopyBuf);
      if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclrtFreeHost failed err[%d]", static_cast<int32_t>(ret));
      }
    }
  }
  INFO_LOG("parse aclFloat16 output data success");
  return;
}

Result SampleRes50ImagenetClassification::Process() {
  string testFile[] = {"../data/dog1_1024_683.bin", "../data/dog2_1024_683.bin"};
  void *picDevBuffer = nullptr;
  size_t devBufferSize = aclmdlGetInputSizeByIndex(modelDesc_->GetModelDesc(), 0);
  aclError aclRet = aclrtMalloc(&picDevBuffer, devBufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (aclRet != ACL_SUCCESS) {
    ERROR_LOG("malloc device buffer failed. size is %zu, errorCode is %d", devBufferSize, static_cast<int32_t>(aclRet));
    return FAILED;
  }
  modelInput_ = std::make_shared<AclModelInput>(picDevBuffer, devBufferSize, modelDesc_->GetModelDesc());
  modelOutput_ = std::make_shared<AclModelOutput>(modelDesc_->GetModelDesc());

  for (size_t index = 0; index < sizeof(testFile) / sizeof(testFile[0]); ++index) {
    INFO_LOG("start to process file:%s", testFile[index].c_str());
    auto ret = Utils::MemcpyFileToDeviceBuffer(testFile[index], picDevBuffer, devBufferSize, g_isDevice);
    if (ret != SUCCESS) {
      aclrtFree(picDevBuffer);
      ERROR_LOG("memcpy device buffer failed, index is %zu", index);
      return FAILED;
    }
    aclError aclRet = aclmdlExecute(modelId_, modelInput_->GetDataSet(), modelOutput_->GetDataSet());
    if (aclRet != ACL_SUCCESS) {
      return FAILED;
    }
    // 解析aclFloat16格式输出
    OutputModelResult();
  }

  aclrtFree(picDevBuffer);
  return SUCCESS;
}

int main() {
  INFO_LOG("SAMPLE start to execute.");
  {
    SampleRes50ImagenetClassification sampleRes50;
    const char *aclConfigPath = "../src/acl.json";
    Result ret = sampleRes50.InitResource(aclConfigPath);
    if (ret != SUCCESS) {
      ERROR_LOG("SAMPLE NOT PASSED: sample init resource failed.");
      return FAILED;
    }
    ret = sampleRes50.PrepareModel("../model/resnet50.om");
    if (ret != SUCCESS) {
      ERROR_LOG("SAMPLE NOT PASSED: sample prepare model failed.");
      return FAILED;
    }
    ret = sampleRes50.Process();
    if (ret != SUCCESS) {
      ERROR_LOG("SAMPLE NOT PASSED: sample process failed.");
      return FAILED;
    }
  }
  INFO_LOG("SAMPLE PASSED.");
  return SUCCESS;
}
