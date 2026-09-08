#ifndef __HELLO_WORLD__KERNEL_FUN_H__
#define __HELLO_WORLD__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define hello_world hello_world_origin
#include "/mnt/class/26EAIHFSIC001/samples/tutorials/HelloWorldSample/hello_world.cpp"

#undef hello_world
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_hello_world_kernel(
#ifdef ASCENDC_DUMP
GM_ADDR dumpAddr
#endif
) {
#ifdef ASCENDC_DUMP
    InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);
#endif

    hello_world_origin();
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
