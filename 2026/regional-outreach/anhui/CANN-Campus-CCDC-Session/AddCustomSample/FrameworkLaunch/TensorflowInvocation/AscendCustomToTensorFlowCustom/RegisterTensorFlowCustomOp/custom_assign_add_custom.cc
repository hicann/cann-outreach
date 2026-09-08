/**
 * @file custom_assign_add_custom.cc
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

// 注册TensorFlow自定义算子
REGISTER_OP("AddCustom")                                    // TensorFlow自定义算子名称
    .Input("x: T")                                          // 输入tensor x
    .Input("y: T")                                          // 输入tensor y
    .Output("z: T")                                         // 输出tensor z
    .Attr("T: {half}")                                      // 属性T，支持half数据类型
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn); // 设置shape函数，用于推断输出tensor的shape，BroadcastBinaryOpShapeFn函数用于处理输入、输出tensor的shape相同的情况


// TensorFlow自定义算子的CPU实现
class AddCustomOp : public OpKernel {
public:
    explicit AddCustomOp(OpKernelConstruction* context) : OpKernel(context) {}
    // 当前算子不支持CPU设备，实现该函数以抛出异常，提示该算子不支持CPU设备
    void Compute(OpKernelContext* context) override {
        OP_REQUIRES(context, false, errors::Unimplemented("AddCustomOp is not supported on CPU"));
    }
};

// 注册TensorFlow自定义算子的CPU实现
REGISTER_KERNEL_BUILDER(Name("AddCustom").Device(DEVICE_CPU), AddCustomOp);