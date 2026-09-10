### The First Cicle
• 代码已完成并通过真实端到端测试。

  - Llama-2-13B、TP=4、FP16、XFormers 成功启动。
  - 128-token prompt、drop_ratio=0.25、生成 3 token，API 返回 HTTP 200。
  - 两轮 decode 各重算 32 token，指标为 vllm:ellm_recompute_tokens_total = 64。
  - 配置指标正确暴露 drop_ratio=0.25、max_recompute_tokens=2048、overlap_mode=streams。
  - CUDA 重算前缀算子与完整 KV attention 数值对比误差为 0.0。
  - Python 编译、block manager 断言、配置断言和 git diff --check 均通过。
  - 测试完成后已停止服务，四张 GPU 均恢复为 0 MiB 使用。

  主要入口位于：

  - 静态参数：vllm/vllm/engine/arg_utils.py:299
  - KV 前缀丢弃：vllm/vllm/core/block_manager_v1.py:264
  - 重算预算调度：vllm/vllm/core/scheduler.py:429
  - Decode 输入重建：vllm/vllm/worker/model_runner.py:444
  - XFormers 双流执行：vllm/vllm/attention/backends/xformers.py:275
  - 混合重算/驻留 KV CUDA kernel：vllm/csrc/attention/attention_kernels.cu:852
  - 实验配置：.vscode/launch.json:5

  当前边界需要明确：streams 已实现两个 CUDA stream 的并发，decode kernel 也能同时读取重算前缀和驻留后缀；但它还不是 supplement.md M5 描述的单 kernel 内 K1/K2 warp
  分区，因此暂时没有伪造一个实际不生效的 Stream A/B 比例参数。真正做 0.5:0.5、0.75:0.25 比例实验，需要下一阶段实现融合 kernel 变体。

代码具体过程：

  1. vllm/vllm/worker/model_runner.py:505 数 block table 开头连续的 -1，得到 num_dropped_blocks，然后取原序列最前面的 token IDs 作为重算输入。
  2. vllm/vllm/model_executor/models/llama.py:284 重新 embedding 前缀 token，并把它们放在当前 decode token 前面。之后整个组合 tensor 依次通过每一层。
  3. 每一层的 QKV projection 会生成：
      - 丢弃前缀的重算 Q/K/V；
      - 当前 decode token 的 Q/K/V。

  4. vllm/vllm/attention/backends/xformers.py:275 对前缀执行 causal xFormers attention。这个结果不是当前 token 的 decode 结果，而是为了生成下一层所需的 prefix
     hidden states。

  5. 同时，当前 token 的 decode query 进入 paged_attention_v1_with_recomputed：
      - token_idx < dropped_len：从连续的 recomputed_k/recomputed_v 读取。
      - token_idx >= dropped_len：从正常 paged KV cache 读取。
      - 然后统一计算 QK、softmax 和 softmax·V。

  数据源切换分别在 vllm/csrc/attention/attention_kernels.cu:224 和 vllm/csrc/attention/attention_kernels.cu:358。

  多请求 batch 使用 recompute_start_locs 定位每个请求在扁平重算 K/V tensor 中的起点：

  recomputed_token_idx =
      recompute_start_locs[seq_idx] + token_idx

  6. 前缀和 decode 的 attention 输出重新合并，继续经过 output projection、MLP，形成下一层输入。最后一层结束后才丢掉 prefix 部分，只保留 decode token 输出进入 LM
     head。

  目前 streams 模式会让“前缀 xFormers attention”和“decode PagedAttention”运行在两个 CUDA stream 上，但它们仍是两个独立 kernel 路径。它还不是 supplement.md 所述的
  单 kernel K1/K2 warp 分区融合，也没有真正的 Stream A/B 比例参数。