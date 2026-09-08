// 1. 包含Ascend C编程所需的核心头文件，提供了核函数编写、内存操作、队列等所有基础API。
#include "kernel_operator.h"

// 2. 使用Ascend C的命名空间，这样可以直接使用Tensor、Que等常用类型，无需前缀。
using namespace AscendC;

/**
 * 3. 核函数（Kernel Function）定义
 *    这是整个程序的入口点，将在NPU（AI Core）上执行。
 *    extern "C"：防止C++编译器对函数名进行修饰（Name Mangling），确保从Host侧能通过C语言规则找到这个函数。
 *    __global__：Ascend C的修饰符，表明这是一个“入口函数”，可以被Host侧代码调用并启动。
 *    __aicore__：明确指定该函数在AI Core（计算核心）上运行，而不是在其他处理单元。
 *    函数名：hello_world，通过<<<>>>语法从Host侧启动。
 *    返回值：void，核函数通常不返回值，而是通过指针修改数据或直接输出（如这里的打印）。
 */
extern "C" __global__ __aicore__ void hello_world()
{
    /**
     * 4. 核心执行逻辑
     *    在NPU上调用AscendC命名空间下的printf函数。
     *    此函数用于在NPU侧打印信息，通常用于调试或验证环境是否正常工作。
     *    注意：NPU上的打印是异步的，实际输出可能稍有延迟，且频繁使用会影响性能。
     */
    AscendC::printf("Hello World!!!\n");
}

/**
 * 5. Host侧封装函数（Wrapper Function）
 *    这个函数运行在CPU（Host）侧，用于封装对NPU核函数的启动调用。
 *    它提供了一种更简洁、类型安全的方式来启动核函数，使得在main函数中调用更方便。
 *    
 *    @param blockDim 启动的AI Core数量，即并行执行的核数。
 *    @param stream  AscendCL的Stream（流）句柄，用于管理异步任务的执行顺序。
 *                 同一个流中的任务按顺序执行，不同流可以并行。
 */
void hello_world_do(uint32_t blockDim, void* stream)
{
    /**
     * 6. 内核调用符（Kernel Launch Operator）
     *    这是启动核函数的关键语法：hello_world <<< blockDim, nullptr, stream >>>();
     *    - 第一个参数 blockDim：指定启动的AI Core数量（即并行度）。
     *    - 第二个参数 nullptr：预留参数，用于设置额外的运行时配置，通常传nullptr。
     *    - 第三个参数 stream：指定任务在哪个流上执行，用于实现异步操作。
     *    执行此行代码后，Host侧程序会继续执行，而NPU会异步开始运行hello_world核函数。
     */
    hello_world<<<blockDim, nullptr, stream>>>();
}