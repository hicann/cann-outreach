#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <cstring>

#include "acl/acl.h"

// 读取二进制文件
static void* ReadBinFile(const std::string& filePath, uint32_t& fileSize) {
    std::ifstream file(filePath, std::ifstream::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return nullptr;
    }
    file.seekg(0, file.end);
    fileSize = file.tellg();
    file.seekg(0, file.beg);
    uint8_t* buffer = new uint8_t[fileSize];
    file.read(reinterpret_cast<char*>(buffer), fileSize);
    file.close();
    return buffer;
}

// 打印 Top5
static void PrintTop5(const float* outputData, int size) {
    if (outputData == nullptr || size <= 0) return;
    std::vector<std::pair<float, int>> probIndex;
    probIndex.reserve(size);
    for (int i = 0; i < size; ++i) {
        probIndex.emplace_back(outputData[i], i);
    }
    std::partial_sort(probIndex.begin(), probIndex.begin() + 5, probIndex.end(),
                      std::greater<std::pair<float, int>>());
    std::cout << "=== Top 5 ===" << std::endl;
    for (int i = 0; i < 5 && i < static_cast<int>(probIndex.size()); ++i) {
        std::cout << "Index: " << probIndex[i].second
                  << ", Probability: " << probIndex[i].first << std::endl;
    }
}

class ResNet50Infer {
public:
    ResNet50Infer() : deviceId_(0), context_(nullptr), stream_(nullptr), modelId_(0),
                      modelWorkPtr_(nullptr), modelWeightPtr_(nullptr),
                      inputBuffer_(nullptr), outputBuffer_(nullptr),
                      inputDataset_(nullptr), outputDataset_(nullptr), modelDesc_(nullptr),
                      isInited_(false) {}

    ~ResNet50Infer() { Finalize(); }

    // 初始化：加载 om 模型，读取 bin 输入
    aclError Init(const std::string& modelPath, const std::string& binPath) {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclInit failed, errorCode = " << ret << std::endl;
            return ret;
        }
        ret = aclrtSetDevice(deviceId_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtSetDevice failed, errorCode = " << ret << std::endl;
            return ret;
        }
        ret = aclrtCreateContext(&context_, deviceId_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtCreateContext failed, errorCode = " << ret << std::endl;
            return ret;
        }
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtCreateStream failed, errorCode = " << ret << std::endl;
            return ret;
        }

        ret = LoadModel(modelPath);
        if (ret != ACL_SUCCESS) return ret;

        ret = LoadInputFromBin(binPath);
        if (ret != ACL_SUCCESS) return ret;

        ret = PrepareInputOutput();
        if (ret != ACL_SUCCESS) return ret;

        isInited_ = true;
        return ACL_SUCCESS;
    }

    // 执行推理，返回 Softmax 后的概率向量
    std::vector<float> Infer() {
        std::vector<float> result;
        if (!isInited_) {
            std::cerr << "Model not initialized." << std::endl;
            return result;
        }

        aclError ret = aclmdlExecute(modelId_, inputDataset_, outputDataset_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlExecute failed, errorCode = " << ret << std::endl;
            return result;
        }

        size_t outputNum = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
        size_t numClasses = outputNum / sizeof(float);
        float* outputHost = new float[numClasses];
        ret = aclrtMemcpy(outputHost, outputNum, outputBuffer_, outputNum, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy output failed, errorCode = " << ret << std::endl;
            delete[] outputHost;
            return result;
        }

        std::vector<float> logits(outputHost, outputHost + numClasses);
        result = Softmax(logits);
        delete[] outputHost;
        return result;
    }

    void Run(const std::string& modelPath, const std::string& binPath) {
        aclError ret = Init(modelPath, binPath);
        if (ret != ACL_SUCCESS) {
            std::cerr << "Init failed, errorCode = " << ret << std::endl;
            return;
        }
        std::vector<float> result = Infer();
        if (result.empty()) {
            std::cerr << "Infer failed." << std::endl;
            return;
        }
        PrintTop5(result.data(), result.size());
    }

private:
    aclError LoadModel(const std::string& modelPath) {
        uint32_t modelSize = 0;
        void* modelBuf = ReadBinFile(modelPath, modelSize);
        if (!modelBuf) {
            std::cerr << "Read model file failed." << std::endl;
            return ACL_ERROR_FAILURE;
        }

        aclError ret = aclmdlQuerySizeFromMem(modelBuf, modelSize, &modelWorkSize_, &modelWeightSize_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlQuerySizeFromMem failed, errorCode = " << ret << std::endl;
            delete[] (uint8_t*)modelBuf;
            return ret;
        }

        ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc modelWorkPtr_ failed, errorCode = " << ret << std::endl;
            delete[] (uint8_t*)modelBuf;
            return ret;
        }
        ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc modelWeightPtr_ failed, errorCode = " << ret << std::endl;
            aclrtFree(modelWorkPtr_);
            delete[] (uint8_t*)modelBuf;
            return ret;
        }

        ret = aclmdlLoadFromFileWithMem(modelPath.c_str(), &modelId_,
                                        modelWorkPtr_, modelWorkSize_,
                                        modelWeightPtr_, modelWeightSize_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlLoadFromFileWithMem failed, errorCode = " << ret << std::endl;
            aclrtFree(modelWorkPtr_);
            aclrtFree(modelWeightPtr_);
            delete[] (uint8_t*)modelBuf;
            return ret;
        }
        delete[] (uint8_t*)modelBuf;

        modelDesc_ = aclmdlCreateDesc();
        ret = aclmdlGetDesc(modelDesc_, modelId_);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclmdlGetDesc failed, errorCode = " << ret << std::endl;
            return ret;
        }
        return ACL_SUCCESS;
    }

    aclError LoadInputFromBin(const std::string& binPath) {
        uint32_t binSize = 0;
        void* binBuf = ReadBinFile(binPath, binSize);
        if (!binBuf) {
            std::cerr << "Read input bin file failed." << std::endl;
            return ACL_ERROR_FAILURE;
        }

        size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
        if (binSize != modelInputSize) {
            std::cerr << "Bin file size mismatch: got " << binSize
                      << ", expected " << modelInputSize << std::endl;
            delete[] (uint8_t*)binBuf;
            return ACL_ERROR_FAILURE;
        }

        aclError ret = aclrtMalloc(&inputBuffer_, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc inputBuffer_ failed, errorCode = " << ret << std::endl;
            delete[] (uint8_t*)binBuf;
            return ret;
        }

        ret = aclrtMemcpy(inputBuffer_, modelInputSize, binBuf, binSize, ACL_MEMCPY_HOST_TO_DEVICE);
        delete[] (uint8_t*)binBuf;
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMemcpy inputBuffer_ failed, errorCode = " << ret << std::endl;
            aclrtFree(inputBuffer_);
            inputBuffer_ = nullptr;
            return ret;
        }
        return ACL_SUCCESS;
    }

    aclError PrepareInputOutput() {
        inputDataset_ = aclmdlCreateDataset();
        if (!inputDataset_) {
            std::cerr << "aclmdlCreateDataset inputDataset_ failed" << std::endl;
            return ACL_ERROR_FAILURE;
        }
        size_t inputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
        aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer_, inputSize);
        if (!inputData) {
            std::cerr << "aclCreateDataBuffer inputData failed" << std::endl;
            aclmdlDestroyDataset(inputDataset_);
            inputDataset_ = nullptr;
            return ACL_ERROR_FAILURE;
        }
        aclmdlAddDatasetBuffer(inputDataset_, inputData);

        outputDataset_ = aclmdlCreateDataset();
        if (!outputDataset_) {
            std::cerr << "aclmdlCreateDataset outputDataset_ failed" << std::endl;
            return ACL_ERROR_FAILURE;
        }
        size_t outputSize = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
        aclError ret = aclrtMalloc(&outputBuffer_, outputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            std::cerr << "aclrtMalloc outputBuffer_ failed, errorCode = " << ret << std::endl;
            return ret;
        }
        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer_, outputSize);
        if (!outputData) {
            std::cerr << "aclCreateDataBuffer outputData failed" << std::endl;
            aclrtFree(outputBuffer_);
            outputBuffer_ = nullptr;
            aclmdlDestroyDataset(outputDataset_);
            outputDataset_ = nullptr;
            return ACL_ERROR_FAILURE;
        }
        aclmdlAddDatasetBuffer(outputDataset_, outputData);
        return ACL_SUCCESS;
    }

    void Finalize() {
        if (inputDataset_) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(inputDataset_); ++i) {
                aclDestroyDataBuffer(aclmdlGetDatasetBuffer(inputDataset_, i));
            }
            aclmdlDestroyDataset(inputDataset_);
            inputDataset_ = nullptr;
        }
        if (outputDataset_) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputDataset_); ++i) {
                aclDestroyDataBuffer(aclmdlGetDatasetBuffer(outputDataset_, i));
            }
            aclmdlDestroyDataset(outputDataset_);
            outputDataset_ = nullptr;
        }
        if (modelDesc_) { aclmdlDestroyDesc(modelDesc_); modelDesc_ = nullptr; }
        if (modelId_) { aclmdlUnload(modelId_); modelId_ = 0; }
        if (inputBuffer_) { aclrtFree(inputBuffer_); inputBuffer_ = nullptr; }
        if (outputBuffer_) { aclrtFree(outputBuffer_); outputBuffer_ = nullptr; }
        if (modelWorkPtr_) { aclrtFree(modelWorkPtr_); modelWorkPtr_ = nullptr; }
        if (modelWeightPtr_) { aclrtFree(modelWeightPtr_); modelWeightPtr_ = nullptr; }
        if (stream_) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (context_) { aclrtDestroyContext(context_); context_ = nullptr; }
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

    std::vector<float> Softmax(const std::vector<float>& input) {
        std::vector<float> output(input.size());
        float maxVal = *std::max_element(input.begin(), input.end());
        float sum = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::exp(input[i] - maxVal);
            sum += output[i];
        }
        for (float& v : output) v /= sum;
        return output;
    }

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    uint32_t modelId_;
    void* modelWorkPtr_;
    void* modelWeightPtr_;
    size_t modelWorkSize_;
    size_t modelWeightSize_;
    void* inputBuffer_;
    void* outputBuffer_;
    aclmdlDataset* inputDataset_;
    aclmdlDataset* outputDataset_;
    aclmdlDesc* modelDesc_;
    bool isInited_;
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.om> <input.bin>" << std::endl;
        return -1;
    }
    ResNet50Infer infer;
    infer.Run(argv[1], argv[2]);
    return 0;
}