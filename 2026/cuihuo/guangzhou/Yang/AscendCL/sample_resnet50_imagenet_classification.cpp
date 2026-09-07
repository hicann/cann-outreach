/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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


Result SampleRes50ImagenetClassification::InitResource(const char *modelPath)
{
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

    // Get run mode.
    // ACL_HOST   : app runs on Host.
    // ACL_DEVICE : app runs on Device.
    aclrtRunMode runMode;

    auto ret = aclrtGetRunMode(&runMode);

    if (ret != ACL_SUCCESS) {
        ERROR_LOG(
            "aclrtGetRunMode failed, errorCode[%d]",
            static_cast<int32_t>(ret)
        );

        return FAILED;
    }

    g_isDevice = (runMode == ACL_DEVICE);

    return SUCCESS;
}


Result SampleRes50ImagenetClassification::PrepareModel(const char *modelPath)
{
    // Query model work memory and weight memory.
    size_t modelWorkSize = 0;
    size_t modelWeightSize = 0;

    aclError ret = aclmdlQuerySize(
        modelPath,
        &modelWorkSize,
        &modelWeightSize
    );

    if (ret != ACL_SUCCESS) {
        ERROR_LOG(
            "aclmdlQuerySize failed, errorCode[%d]",
            static_cast<int32_t>(ret)
        );

        return FAILED;
    }

    modelWork_ =
        std::make_shared<AclModelWork>(modelWorkSize);

    modelWeight_ =
        std::make_shared<AclModelWeight>(modelWeightSize);

    // Load model.
    ret = aclmdlLoadFromFileWithMem(
        modelPath,
        &modelId_,
        modelWork_->GetModelWorkPtr(),
        modelWork_->GetModelWorkSize(),
        modelWeight_->GetModelWeightPrt(),
        modelWeight_->GetModelWeightSize()
    );

    if (ret != ACL_SUCCESS) {
        ERROR_LOG(
            "aclmdlLoadFromFileWithMem failed, errorCode[%d]",
            static_cast<int32_t>(ret)
        );

        return FAILED;
    }

    INFO_LOG(
        "load model %s success.",
        modelPath
    );

    // Create ModelDesc.
    modelDesc_ =
        std::make_shared<AclModelDesc>(modelId_);

    return SUCCESS;
}


/**
 * Parse model output.
 *
 * Important:
 * The current OM model output is FP16.
 *
 * So the output buffer must be interpreted as aclFloat16,
 * NOT float.
 *
 * Then every FP16 element is converted to FP32 by
 * aclFloat16ToFloat() before sorting and printing.
 */
void SampleRes50ImagenetClassification::OutputModelResult()
{
    size_t outputBufferNum =
        aclmdlGetDatasetNumBuffers(
            modelOutput_->GetDataSet()
        );

    for (size_t i = 0;
         i < outputBufferNum;
         ++i) {

        // Get model output buffer.
        aclDataBuffer *dataBuffer =
            aclmdlGetDatasetBuffer(
                modelOutput_->GetDataSet(),
                i
            );

        if (dataBuffer == nullptr) {
            ERROR_LOG(
                "aclmdlGetDatasetBuffer failed, index[%zu]",
                i
            );

            return;
        }

        // Device output address.
        void *data =
            aclGetDataBufferAddr(dataBuffer);

        uint32_t len =
            aclGetDataBufferSizeV2(dataBuffer);

        if (data == nullptr || len == 0) {
            ERROR_LOG(
                "model output buffer is invalid, index[%zu], len[%u]",
                i,
                len
            );

            return;
        }

        INFO_LOG(
            "output buffer[%zu] size[%u] bytes",
            i,
            len
        );

        void *outHostData = nullptr;

        // outputData points to the actual memory that CPU can read.
        void *outputData = data;

        aclError ret = ACL_SUCCESS;

        /*
         * If app runs on Host:
         *
         * Model output is in NPU Device memory,
         * so copy it to Host memory first.
         */
        if (!g_isDevice) {

            ret = aclrtMallocHost(
                &outHostData,
                len
            );

            if (ret != ACL_SUCCESS) {
                ERROR_LOG(
                    "aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                    len,
                    static_cast<int32_t>(ret)
                );

                return;
            }

            ret = aclrtMemcpy(
                outHostData,
                len,
                data,
                len,
                ACL_MEMCPY_DEVICE_TO_HOST
            );

            if (ret != ACL_SUCCESS) {
                ERROR_LOG(
                    "aclrtMemcpy DEVICE_TO_HOST failed, errorCode[%d]",
                    static_cast<int32_t>(ret)
                );

                (void)aclrtFreeHost(outHostData);

                return;
            }

            outputData = outHostData;
        }

        /*
         * ============================================================
         *                 FP16 output parsing
         * ============================================================
         *
         * Previous code:
         *
         *     float *outData = reinterpret_cast<float *>(outputData);
         *
         * This is WRONG for FP16 model output.
         *
         * FP32:
         *     sizeof(float) = 4 bytes
         *
         * FP16:
         *     sizeof(aclFloat16) = 2 bytes
         *
         * For example, ResNet50 has 1000 output classes.
         *
         * FP16 output buffer:
         *
         *     1000 * 2 = 2000 bytes
         *
         * If interpreted as float:
         *
         *     2000 / 4 = 500 elements
         *
         * That means two FP16 values are incorrectly combined
         * into one FP32 value.
         *
         * Both confidence value AND class index become incorrect.
         * ============================================================
         */

        aclFloat16 *fp16Data =
            reinterpret_cast<aclFloat16 *>(
                outputData
            );

        size_t elementCount =
            len / sizeof(aclFloat16);

        INFO_LOG(
            "output element count[%zu], output type[FP16]",
            elementCount
        );

        /*
         * results:
         *
         * first  = confidence value
         * second = ImageNet class index
         */
        std::vector<
            std::pair<float, unsigned int>
        > results;

        results.reserve(elementCount);

        /*
         * Convert every FP16 value to FP32.
         */
        for (size_t j = 0;
             j < elementCount;
             ++j) {

            float value =
                aclFloat16ToFloat(
                    fp16Data[j]
                );

            results.emplace_back(
                value,
                static_cast<unsigned int>(j)
            );
        }

        /*
         * Sort confidence from high to low.
         */
        std::sort(
            results.begin(),
            results.end(),
            [](
                const std::pair<float, unsigned int> &a,
                const std::pair<float, unsigned int> &b
            ) {
                return a.first > b.first;
            }
        );

        /*
         * Print Top5.
         */
        size_t topK =
            std::min<size_t>(
                5,
                results.size()
            );

        for (size_t j = 0;
             j < topK;
             ++j) {

            INFO_LOG(
                "top %zu: index[%u] value[%f]",
                j + 1,
                results[j].second,
                results[j].first
            );
        }

        /*
         * Release Host output memory.
         */
        if (!g_isDevice) {

            ret =
                aclrtFreeHost(
                    outHostData
                );

            if (ret != ACL_SUCCESS) {
                ERROR_LOG(
                    "aclrtFreeHost failed, errorCode[%d]",
                    static_cast<int32_t>(ret)
                );

                return;
            }
        }
    }

    INFO_LOG(
        "output data success"
    );
}


Result SampleRes50ImagenetClassification::Process()
{
    string testFile[] = {
        "../data/dog1_1024_683.bin",
        "../data/dog2_1024_683.bin"
    };

    void *picDevBuffer = nullptr;

    /*
     * Get the size of model input 0.
     */
    size_t devBufferSize =
        aclmdlGetInputSizeByIndex(
            modelDesc_->GetModelDesc(),
            0
        );

    /*
     * Allocate input device memory.
     */
    aclError aclRet =
        aclrtMalloc(
            &picDevBuffer,
            devBufferSize,
            ACL_MEM_MALLOC_HUGE_FIRST
        );

    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG(
            "malloc device buffer failed. "
            "size is %zu, errorCode is %d",
            devBufferSize,
            static_cast<int32_t>(aclRet)
        );

        return FAILED;
    }

    /*
     * Create model input dataset.
     */
    modelInput_ =
        std::make_shared<AclModelInput>(
            picDevBuffer,
            devBufferSize,
            modelDesc_->GetModelDesc()
        );

    /*
     * Create model output dataset.
     */
    modelOutput_ =
        std::make_shared<AclModelOutput>(
            modelDesc_->GetModelDesc()
        );

    /*
     * Run inference for each test image.
     */
    for (size_t index = 0;
         index <
             sizeof(testFile) /
             sizeof(testFile[0]);
         ++index) {

        INFO_LOG(
            "start to process file:%s",
            testFile[index].c_str()
        );

        /*
         * Copy image binary data into
         * device input buffer.
         */
        auto ret =
            Utils::MemcpyFileToDeviceBuffer(
                testFile[index],
                picDevBuffer,
                devBufferSize,
                g_isDevice
            );

        if (ret != SUCCESS) {

            (void)aclrtFree(
                picDevBuffer
            );

            ERROR_LOG(
                "memcpy device buffer failed, index is %zu",
                index
            );

            return FAILED;
        }

        /*
         * Execute model inference.
         */
        aclRet =
            aclmdlExecute(
                modelId_,
                modelInput_->GetDataSet(),
                modelOutput_->GetDataSet()
            );

        if (aclRet != ACL_SUCCESS) {

            ERROR_LOG(
                "aclmdlExecute failed, errorCode[%d]",
                static_cast<int32_t>(aclRet)
            );

            (void)aclrtFree(
                picDevBuffer
            );

            return FAILED;
        }

        /*
         * Print Top5 confidence values and indexes.
         *
         * The model output is FP16,
         * OutputModelResult() converts FP16 -> FP32
         * before sorting.
         */
        OutputModelResult();
    }

    /*
     * Release model input device memory.
     */
    aclRet =
        aclrtFree(
            picDevBuffer
        );

    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG(
            "aclrtFree failed, errorCode[%d]",
            static_cast<int32_t>(aclRet)
        );

        return FAILED;
    }

    return SUCCESS;
}


int main()
{
    INFO_LOG(
        "SAMPLE start to execute."
    );

    /*
     * To better demonstrate the core usage
     * of ACL interfaces, this sample encapsulates
     * resource management in
     * SampleRes50ImagenetClassification.
     *
     * Resource release relies on destructors.
     *
     * Therefore the following scope guarantees
     * that all resources are destroyed before
     * the process exits.
     */
    {
        SampleRes50ImagenetClassification
            sampleRes50;

        const char *aclConfigPath =
            "../src/acl.json";

        Result ret =
            sampleRes50.InitResource(
                aclConfigPath
            );

        if (ret != SUCCESS) {
            ERROR_LOG(
                "SAMPLE NOT PASSED: "
                "sample init resource failed."
            );

            return FAILED;
        }

        /*
         * Load FP16 output OM model.
         */
        ret =
            sampleRes50.PrepareModel(
                "../model/resnet50.om"
            );

        if (ret != SUCCESS) {
            ERROR_LOG(
                "SAMPLE NOT PASSED: "
                "sample prepare model failed."
            );

            return FAILED;
        }

        /*
         * Execute inference.
         */
        ret =
            sampleRes50.Process();

        if (ret != SUCCESS) {
            ERROR_LOG(
                "SAMPLE NOT PASSED: "
                "sample process failed."
            );

            return FAILED;
        }
    }

    INFO_LOG(
        "SAMPLE PASSED."
    );

    return SUCCESS;
}