// relu_proto.h — 图模式适配
#ifndef RELU_PROTO_H
#define RELU_PROTO_H
// PROTO_OP(Relu)
//     .Input("x", {DT_FLOAT16})
//     .Output("y", {DT_FLOAT16})
//     .InferShape([](const auto& ctx) {
//         ctx.SetOutputShape(0, ctx.GetInputShape(0));
//     });
#endif
