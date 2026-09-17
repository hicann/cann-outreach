/**
 * @file data_utils.h
 * @brief Host 侧数据文件读写辅助函数（仅 Host 侧使用，Kernel 侧不包含该头文件）
 */

#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

/**
 * @brief 判断文件是否存在（用于可选文件，如真值文件，避免误报错误日志）
 */
inline bool FileExists(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary);
    return file.good();
}

/**
 * @brief 获取文件大小（单位：Byte），文件不存在或打开失败时返回 0
 */
inline size_t GetFileSize(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("[ERROR] open file %s failed\n", filePath.c_str());
        return 0;
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    file.close();
    return fileSize;
}

/**
 * @brief 从文件读取数据到 buffer
 */
inline void ReadFile(const std::string& filePath, size_t fileSize, void* buffer, size_t bufferSize)
{
    if (bufferSize < fileSize) {
        printf("[ERROR] buffer size %zu is smaller than file size %zu, file: %s\n",
               bufferSize, fileSize, filePath.c_str());
        return;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("[ERROR] open file %s failed\n", filePath.c_str());
        return;
    }
    file.read(static_cast<char*>(buffer), static_cast<std::streamsize>(fileSize));
    file.close();
    printf("[INFO] read file %s, size = %zu Byte\n", filePath.c_str(), fileSize);
}

/**
 * @brief 把 buffer 中的数据写入文件
 */
inline void WriteFile(const std::string& filePath, void* buffer, size_t size)
{
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("[ERROR] open file %s failed\n", filePath.c_str());
        return;
    }
    file.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(size));
    file.close();
    printf("[INFO] write file %s, size = %zu Byte\n", filePath.c_str(), size);
}

#endif  // DATA_UTILS_H
