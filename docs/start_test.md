按本地的 ShareGPT 数据集处理。关键点：所有 vLLM 命令从 /workspace/ellm_workspace/vllm 目录执
行，否则外层 vllm/ 会遮蔽 Python 包。我也已经给 .venv 补装了 aiohttp，并加了一个无依赖可视化脚本：scripts/
vllm_benchmark_report.py。

1. 启动 vLLM 服务

cd /workspace/ellm_workspace/vllm
source ../.venv/bin/activate
mkdir -p results logs

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model ../model/Llama-2-13b-hf \
  --tokenizer ../model/Llama-2-13b-hf \
  --served-model-name llama2-13b \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --swap-space 16 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --disable-log-requests \
  2>&1 | tee logs/vllm_server.log
(

    运行情况：/tmp/ray/... is over 95% full
     这是磁盘告警。当前 / 分区 97% 已用，Ray 默认用 /tmp/ray。小测试能跑，但压测时可能出问题。建议重启
     服务前加：

  mkdir -p /workspace/ellm_workspace/ray_tmp
  export RAY_TMPDIR=/workspace/ellm_workspace/ray_tmp
)


ellm版本测试：
```bash
python -m vllm.entrypoints.openai.api_server \
  --host localhost \
  --port 8000 \
  --model ../model/Llama-2-13b-hf \
  --tokenizer ../model/Llama-2-13b-hf \
  --served-model-name llama2-13b \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --swap-space 16 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  --disable-log-requests \
  --ellm-drop-ratio 0.01 \
  --ellm-max-recompute-tokens 8192 \
  --ellm-overlap-mode streams
```

等看到服务启动后，另开一个终端检查:

cd /workspace/ellm_workspace/vllm
source ../.venv/bin/activate

curl http://localhost:8000/health
curl http://localhost:8000/v1/models

2. 跑 ShareGPT 小规模压测

python benchmarks/benchmark_serving.py \
  --backend vllm \
  --host localhost \
  --port 8000 \
  --endpoint /v1/completions \
  --model llama2-13b \
  --tokenizer ../model/Llama-2-13b-hf \
  --dataset-name sharegpt \
  --dataset-path ../dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 800 \
  --sharegpt-output-len 512 \
  --request-rate 2 \
  --save-result \
  --metrics-interval 1 \
  --result-dir results \
  --metadata tp=4 max_model_len=2048

先用 32 条确认能跑通；之后再改 --num-prompts 256/1000，或把 --request-rate 调成 0.5 1 2 4 inf 观察压力变化。


3. 可视化 benchmark 结果

python ../scripts/vllm_benchmark_report.py results/*.json \
  -o results/sharegpt_report.html

然后打开：

/results/sharegpt_report.html

报告里会看到请求吞吐、输入/输出 token 吞吐、TTFT、TPOT、GPU/CPU KV cache
使用率、输入输出长度分布和生成文本预览。使用 `--save-result --backend vllm` 时，benchmark
默认从 `http://<host>:<port>/metrics` 每秒采样一次；`--metrics-interval` 可以调整间隔，
`--metrics-url` 可以覆盖地址，`--disable-server-metrics` 可以关闭采样。

旧的 benchmark JSON 没有 `server_metrics` 字段，因此重新生成 HTML 也只能显示
`Not collected for this run`；要获得准确的逐实验 KV cache 曲线，需要重新运行实验。

运行时也可以直接观察服务端日志里的这些指标：

Avg prompt throughput
Avg generation throughput
Running / Swapped / Pending
GPU KV cache usage
CPU KV cache usage

原始 Prometheus 指标在这里：

curl http://localhost:8000/metrics

─ Worked for 8m 08s 