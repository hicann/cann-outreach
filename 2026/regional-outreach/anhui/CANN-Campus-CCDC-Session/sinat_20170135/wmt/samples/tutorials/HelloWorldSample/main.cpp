// 1. 包含AscendCL（Ascend Computing Language）头文件，提供设备管理、流管理、内存管理等运行时API
#include "acl/acl.h"

// 2. 外部声明核函数启动封装函数，该函数在hello_world.cpp中实现，用于启动NPU上的核函数
extern void hello_world_do(uint32_t coreDim, void* stream);

// 3. 主函数入口，argc为命令行参数个数，argv为命令行参数数组
int32_t main(int argc, char const *argv[])
{
    // 4. AscendCL初始化，nullptr表示使用默认配置；必须先调用才能使用其他ACL API
    aclInit(nullptr);
    
    // 5. 声明设备ID变量，0表示使用第一张NPU卡（如Atlas 300I Pro）
    int32_t deviceId = 0;
    
    // 6. 显式指定并启用NPU设备，后续操作都将在该设备上执行
    aclrtSetDevice(deviceId);
    
    // 7. 声明流（Stream）句柄，初始化为空指针
    aclrtStream stream = nullptr;
    
    // 8. 创建一个流，用于管理异步任务的执行顺序；同一个流中的任务按顺序执行
    aclrtCreateStream(&stream);

    // 9. 定义常量blockDim并设置为8，表示启动8个AI Core并行执行核函数（可根据实际需求调整核数）
    constexpr uint32_t blockDim = 8;
    
    // 10. 调用封装函数启动核函数，该函数内部使用<<<>>>语法将hello_world核函数发射到NPU上执行
    hello_world_do(blockDim, stream);
    
    // 11. 同步等待流中的所有任务完成，确保核函数执行完毕后才继续执行后续代码（阻塞调用）
    aclrtSynchronizeStream(stream);
    
    // 12. 销毁流，释放流相关的资源，避免资源泄漏
    aclrtDestroyStream(stream);
    
    // 13. 重置NPU设备，释放设备上的资源，与aclrtSetDevice配对使用
    aclrtResetDevice(deviceId);
    
    // 14. AscendCL去初始化，释放全局资源，与aclInit配对使用
    aclFinalize();
    
    // 15. 返回0表示程序正常退出，非0表示异常退出
    return 0;
}