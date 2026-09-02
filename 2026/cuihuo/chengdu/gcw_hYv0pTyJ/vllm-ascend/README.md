# vllm-ascend 推理部署实践（淬火行动）

本目录为昇腾淬火行动 vllm-ascend 实验任务交付物，包含两个模型的部署验证与性能测试结果。

## 环境
- 环境：DevEnv_09922222（Triton-Ascend / CANN / torch-npu）
- NPU：Ascend 910B3 单卡（HBM 64GB）
- vLLM：0.18.0（vllm-ascend 插件）
- 模型权重来源：/workspace/shared_assets/QuenchingAction/Toolchain/Model/

## 任务1：Qwen2.5-Omni-7B
- 参考：https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen2.5-Omni.html
- 子任务：https://github.com/vllm-project/vllm-ascend/issues/11750
- 启动命令：
  vllm serve /root/user_data/Qwen2.5-Omni-7B --host 0.0.0.0 --port 8000 --served-model-name Qwen-Omni --trust-remote-code --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --no-enable-prefix-caching
- Chat 推理验证：通过（中文对话正常回复）
- 性能测试（vllm bench serve，random 3500→1500，80 prompts，max-concurrency 20）：
  - completed: 80 / failed: 0
  - request_throughput: 0.52 req/s
  - mean TTFT: 1263.76 ms
  - mean TPOT: 24.59 ms
  - median ITL: 22.22 ms
  - 结果文件：benchmark_results_task1/vllm-infqps-concurrency20-Qwen2.5-Omni-7B-20260820-051009.json

## 任务2：Qwen3-VL-8B-Instruct
- 参考：https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen-VL-Dense.html
- 子任务：https://github.com/vllm-project/vllm-ascend/issues/11751
- 启动命令：
  vllm serve /root/user_data/Qwen3-VL-8B-Instruct --dtype bfloat16 --max-model-len 16384 --max-num-batched-tokens 16384 --host 0.0.0.0 --port 8000 --served-model-name Qwen3-VL-8B-Instruct
- 图片理解验证（curl 多模态请求）：通过，识别图片文字 "TONGYI Qwen"（task2_curl_result.json）
- 性能测试（vllm bench serve，random 200 prompts，request-rate 1）：
  - completed: 200 / failed: 0
  - request_throughput: 0.98 req/s
  - mean TTFT: 95.97 ms
  - mean TPOT: 28.22 ms
  - median ITL: 27.81 ms
  - 结果文件：benchmark_results_task2/openai-1.0qps-Qwen3-VL-8B-Instruct-20260820-060223.json

## 备注
- 多模态推理时服务端 --served-model-name 需与请求 model 字段一致（避免 bench 误触发 huggingface 下载）
- 显存管理：切换模型前需 kill 残留的 VLLMEngineCor 进程释放 HBM（pkill -f 'vllm serve' 无法杀子进程）
