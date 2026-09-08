# eLLM 缺失机制补充文档（supplement to AGENT.md）

> 本文档记录 AGENT.md 建模中缺失的 7 个机制，全部来自论文
> *eLLM: Adaptive Attention-State Residency for Memory-Efficient and QoS-Aware LLM Serving*。
> AGENT.md 保持原样；本文档每条机制包含：**论文出处 → 机制原理 → 数学模型 → 决策流程 → 工作示例（Llama-13B / A100 估算）→ vLLM 实现要点**。
>
> ★ 标注处：论文 PDF 提取文本中公式有乱码，已按语义重建，落地前建议对照原文 §4 确认。

---

## 0. 术语表

| 符号 | 含义 |
|---|---|
| B | 每次解码迭代的 batch 大小（请求数），由请求级控制器选择 |
| ρ | uncached-token ratio，batch 级"非驻留比例"，记录在每个 SequenceGroup，可跨迭代更新 |
| s | 单请求当前序列长度（prefill 后初始化，每步解码后 +1） |
| u | 单请求非驻留历史 token 数：u = ρ·s |
| ρ_attn | 每 token 每层注意力状态字节数（MHA: 2·d_h·n_h；GQA/MQA: 2·d_h·n_kv；MLA: latent + aux），见 M5 |
| S_mat | 本迭代需要物化的层集合（非驻留状态所在层） |
| S_swap / S_rec | 分配给"从 host 恢复" / "GPU 上重算"的层集合：S_swap ∪ S_rec = S_mat，S_swap ∩ S_rec = ∅ |
| S_dec | 解码涉及的层集合（通常为全部 L 层） |
| K1 / K2 | 融合核内的两个工作分区：K1 = 重算（compute-bound），K2 = 解码（memory-bound） |
| W1 / W2 | K1 / K2 的工作负载估算（Eq.16） |
| thr / t | 每 block 总线程数 / 分配给 K1 的线程数（K2 得 thr−t） |
| F | token-layer 内存管理中层的分组大小（默认 4） |
| R_attn / T_attn | 持久驻留的注意力状态足迹 / 物化所需的临时工作区（Eq.1 / Eq.15） |
| T_iter | 一次解码迭代的延迟（TPOT 的估计值，Eq.5） |

## 0.1 三处与 AGENT.md 表述不同的地方（实现时请注意，AGENT.md 不改）

1. **kernel_a 的输入**是上一层输出 h_{L-1}(prefix)（L=0 时为 embedding 序列），不是 "last layer 的 hidden_states"。
2. **丢弃规则**：h_L(prefix) 是 kernel_a(L+1) 的输入，kernel_b(L) 解码完不能立刻丢；只保留"一层在途"的 prefix hidden states，等下一层 kernel_a 消费完再释放。
3. **logits/probability 只在最后一层之后**由 LM head 产生（论文：vocabulary projection 在融合路径之外，eLLM 只物化中间状态、不出 logits），不是每层 kernel_b 都输出概率。

---

## M1. 非驻留状态的物化：swap vs recompute 到底怎么决定（§4.3.1, Fig.9/10, Eq.11–14）

### 1.1 机制原理
非驻留状态在 attention 消费前必须被物化（materialize），论文提供两条路径：

- **C 路径（recompute）**：在 GPU 上精确重算。重算 = 重新执行 state-generation 算子，形态类似 prefill，**compute/SM 密集**。注意：重算路径要再生的是 hidden **和** attention 两类状态（Eq.13 明确 "regenerate the hidden and attention states consumed by later decode"）。
- **H 路径（restore / swap）**：把丢弃前备份到 host 内存的 KV 经 host→GPU 链路搬回，**带宽密集**，几乎不耗 SM。

决策目标是：**最小化暴露在解码关键路径上的物化时间**。eLLM 以"层"为单位，把需要物化的层集合 S_mat 划分成 S_swap（恢复）与 S_rec（重算）两部分。这是层级优化器（layer-level optimizer）的工作，与请求级控制器的 (B, ρ) 决策解耦但联动。

前提：走 H 路径要求在"置为非驻留"时先把状态备份到 host（论文实现中给 Llama2-13B / 70B 分别配了 40GB / 160GB host 内存存 host-resident states）；如果丢弃时没备份（纯丢弃），该层只能走 C 路径。**这也说明你的建模"drop 而不备份"= 纯 C 路径，是论文设计空间的一个极端。**

### 1.2 数学模型

**恢复延迟**（Eq.11–12，字节线性模型，Fig.9b 实测 Llama2-13B/A100 传输时间≈线性）：

```
S_swap_bytes = Σ_{l ∈ S_swap}  u · ρ_attn(l)          # 需要搬回的字节数
t_swap(S_swap) = S_swap_bytes / B_swap + c_swap        # B_swap: 实测 host→GPU 有效带宽; c_swap: 常数
```

**重算延迟**（Eq.13，按层累加）：

```
t_rec(S_rec) = Σ_{l ∈ S_rec}  v_rec,state(l) / ρ_eff_rec(l)
# v_rec,state(l) = 再生 layer l 的 hidden+attention 状态所需工作量（≈ u 个 token 的 prefill-like 前向，含 attention 的（超线性）项）
# ρ_eff_rec(l)  = 实测的该层有效吞吐（roofline 方式给出，M6）
# 注意：vocab projection 不计入（只物化中间状态，不出 logits）
```

**层优化问题**（Eq.14，核心决策式）：

```
min_{S_swap, S_rec}   max( t_swap(S_swap), t_rec(S_rec) )
s.t.  S_swap ∪ S_rec = S_mat
      S_swap ∩ S_rec = ∅
      |S_swap| ≤ 3,  |S_rec| ≤ 2
```

- 约束 |S_swap| ≤ 3、|S_rec| ≤ 2 是论文评估中的限制（受临时内存与融合核规模约束），也用作请求级优化时 T_attn 的保守上界（M4）。
- 为什么是 max 而不是 sum：swap 与 recompute 调度为**并发**执行（见 1.4），关键路径取两者较大者；平衡配置（两者时间相近）的暴露物化时间最小，失配会产生"时间气泡"（Fig.9a 的 TR>TS / TR<TS / TR≈TS 三种情形）。

### 1.3 决策流程
1. 请求级控制器给定 batch 与 ρ（M2）→ 得到每个请求的非驻留集合（u = ρ·s 个较老 token，所有层）。
2. 层级优化器对每个请求（或 batch 统一）评估可行的层分配，解 Eq.14，选出 S_swap / S_rec 及重叠与融合配置 θ。
3. 把临时内存（Eq.15，M4）与线程配置（M5）反馈给请求级下一轮迭代（闭环）。

### 1.4 执行：双 CUDA stream + 层间重叠（Fig.10）
- **Stream A**：host→GPU 恢复（异步拷贝，`torch.cuda.Stream` + `torch.cuda.stream`）。
- **Stream B**：融合的 K1∥K2 内核（重算 + 解码，M5）。
- CUDA events 在 attention 消费物化状态前强制层依赖（"reducing serialization without changing execution semantics"）。
- 层间重叠形态：可"重算 i、i+1 层，同时恢复 i+2 层"，或"重算 i 层，同时恢复 i+1、i+2 层"（Fig.9a）；选平衡配置避免时间气泡。

### 1.5 工作示例（Llama-13B / A100，粗估）
常量（估算）：ρ_attn = 2·40·128·2B = 20KB/token/层（fp16，M5）；host→GPU 有效带宽 ≈ 25GB/s（PCIe 4.0×16）；单 token 单层 dense+attention 重算 FLOPs ≈ 0.7 GFLOPs（Llama-13B 规模）。

| 场景 | u | swap 每层 | recompute 每层 | 结论 |
|---|---|---|---|---|
| ShareGPT 短文（s≈222, ρ=0.5） | ≈111 | 111×20KB=2.2MB → ≈0.09ms | 111×0.7G≈78 GFLOPs → 数 ms 级 | **swap 明显更优**（带宽有余量） |
| L-Eval 长文（s≈36k, ρ=0.5） | ≈18k | 18k×20KB=360MB → ≈14ms | 18k×0.7G≈12.6 TFLOPs → 数十 ms 级 | 两者都贵 → **提高驻留比（ρ↓）** |

定性规则：**带宽有余量 → 走 swap；SM 有余量 → 走 recompute（融合吸收空闲 SM）；两者都紧张 → 降 ρ（更多驻留）或全驻留**。这也解释了论文 L-Eval（长文）实测驻留比 53%——物化太贵，宁可多留。

### 1.6 vLLM 实现要点
- eviction 时：把被置为非驻留的 KV 拷到 host（pin memory buffer），而不是直接释放——否则该层永远只能重算。
- 恢复：`torch.cuda.Stream` 上异步 `cudaMemcpyAsync`（H2D）；恢复完成用 CUDA event 通知注意力核。
- 层分配：Eq.14 是个小规模组合问题（层数 ≤ 40，约束 ≤3/≤2），直接枚举即可（论文没有给出算法细节，枚举/贪心都够）。
- 论文实测层级开销：map 查找平均 <0.016ms，线程配置选择 ≈0.03ms。

---

## M2. 请求级控制器：B 和 u(ρ) 到底怎么算（§4.1, §4.2.2, §5）

### 2.1 机制原理
每调度迭代，请求级控制器**联合**选择 batch 大小 B 与 batch 级 uncached-token ratio ρ，目标最大化输出 token 吞吐 B/T_iter，约束：(a) 内存放得下，(b) T_iter ≤ TPOT SLO。ρ 记录在每个 SequenceGroup（batch 内共享），随负载条件跨迭代更新。这是**分层参数化**：请求级只定 (B, ρ)，层级细节（S_swap/S_rec、重叠、融合、线程）由层级优化器在给定 (B, ρ) 后细化，并把结果（临时内存、线程配置、延迟估计）反馈回来。

### 2.2 数学模型

**内存约束**（Eq.4/10）：

```
M_dense + E(x) + R_attn(π) + T_attn(π) + T_MoE(π, x) ≤ M_GPU

M_dense : 非专家模型权重（模型并行后每 GPU 的权重）
E(x)    : MoE 专家权重驻留足迹 (Eq.3) —— dense 模型此项为 0
R_attn  : 持久驻留注意力状态 = Σ_{(s,t,l)∈R(π)} ρ_attn   (Eq.1)
        ≈ (1−ρ) · Σ_i s_i · L · ρ_attn                    # batch 内所有请求的驻留部分
T_attn  : 临时工作区（物化 buffer，Eq.15，见 M4）
T_MoE   : 路由元数据、dispatch/combine buffer、专家暂存工作区（MoE 才有）
```

**TPOT 约束**（Eq.5，迭代延迟模型）：

```
T_iter = Σ_{l=1}^{L} max( t_GPU(l; π, θ; x), t_swap(l) ) + t_launch + t_sync

t_GPU(l) : 层 l 的串行执行：attention + dense/MoE + 通信阶段 + 融合进来的重算
           （每阶段用 roofline：max(FLOPs/有效算力, 字节数/有效带宽)）
t_swap(l): 层 l 可并发执行的 host→GPU 恢复（只对 S_swap 中的层非零）
max(...)  : 只作用于 eLLM 调度为"并发"的操作；层间串行依赖用外层求和保留
t_launch/t_sync: kernel launch 与同步开销（实测常量）
```

**算子体积**（Eq.6–9，喂给 roofline）★：

```
重算 attention 体积:  V_rec,attn = Σ_l v_rec,attn(l),  v_rec,attn(l) ∝ u   # u 个非驻留 token 的 attention 工作
解码 attention 体积:  V_dec,attn = Σ_l v_dec,attn(l),  v_dec,attn(l) ∝ 新 token 数（=1）
dense 体积 (Eq.7):    V_rec,dense = Σ_l (n_dense(l)·u + b_0)
                      V_dec,dense = Σ_l n_dense(l)·1 + 2·d + b_1
                      # n_dense(l): 每 token 的 dense 非注意力算子体积; 2·d: 新 token 的输出投影; b_0,b_1: 杂项/框架开销
MoE (Eq.8–9):         n_{l,e} = Σ_t r_{t,l,e}（路由到专家 e 的 token 数）
                      t_MoE = Σ_{l∈L_MoE} f(n_l, x)  # 实测 grouped-expert 延迟，捕获路由不均与小 microbatch
```

**请求级优化问题**（Eq.10）：

```
max_{B, ρ}   B / T_iter((B, ρ), θ̂; x)
s.t.  T_iter((B, ρ), θ̂; x) ≤ SLO_TPOT
      M_dense + E(x) + R_attn((B,ρ)) + T_attn((B,ρ)) + T_MoE((B,ρ),x) ≤ M_GPU
      0 ≤ ρ ≤ 1,  1 ≤ B ≤ B_max
```

### 2.3 求解流程（u 和 B 到底怎么算出来）
1. **收集信号**：每请求序列长度 s_i、队列长度 q、可用 HBM、注意力足迹 ρ_attn、MoE 专家驻留 x、profiled 硬件常量。
2. **连续松弛**：把 u = ρ·s 松弛为连续变量 ũ = ρ̃·s，用 **SLSQP**（scipy.optimize，论文用 SciPy [45]）解 Eq.10 的连续版本。
3. **整数化**：在候选解 (ρ̃, B̃) 附近检查**相邻可行整数 batch 大小**与 u = round(ρ·s) 诱导的离散驻留方案，选满足内存 + TPOT、估算吞吐最高的配置。
4. **准入**：FCFS（默认）；ρ 记录进每个 SequenceGroup；s 在 prefill 后初始化、每步解码后更新。
5. **闭环**：层级优化器（M1/M4/M5）细化方案后，把精化的 T_attn 与线程配置反馈到下一轮请求级决策。

**u 的确定**：u = round(ρ·s)（单请求非驻留 token 数；驻留部分默认是**较新的 s−u 个 token**，与 AGENT.md 中"丢前 drop_len 个"方向一致，drop_len 即 u）。ρ 不是拍脑袋定的：它由 Eq.10 在"内存约束放开 batch"与"TPOT 约束收紧重算/恢复开销"之间取平衡。

### 2.4 工作示例（Llama-13B / A100，粗估）
- 常量：HBM 80GB，权重 ~26GB（fp16），KV 预算 ≈ 50GB；ρ_attn=20KB/token/层，L=40。
- s=4096 长文：全驻留每请求 = 4096×40×20KB = 3.2GB → B_full ≈ 15；ρ=0.5 → 1.6GB/请求 → B ≈ 31（**翻倍**）。
- 但 TPOT 检查会收窄：ρ=0.5 纯重算时 T_iter 大概率 > SLO_TPOT（重算 ≈ u×L 的 prefill 工作量，M1 示例），控制器就会降 ρ，或把部分层切到 swap、开融合，直到 T_iter ≤ SLO。
- 结论：**B 由内存约束定上界，由 TPOT 约束收窄；ρ 是两者之间的调节旋钮**。

### 2.5 vLLM 实现要点
- 接入点：Scheduler（调度循环），每迭代跑一次优化；论文实测请求级开销 <21ms（90% < 6.85ms），计入端到端指标。
- 队列等待时间不计入 TPOT（属 TTFT 范畴）。
- 若嫌 SLSQP 重，论文的替代路径就是**网格搜索**：对若干 (B, ρ) 候选估 T_iter 与内存，取可行中吞吐最高者——先做这个，再加 SLSQP。

---

## M3. token-layer 内存管理器（§3.2, Fig.7, §6.5）

### 3.1 机制原理
vLLM 的 PagedAttention 把每条序列切成固定大小 token block，逻辑 block 映射到物理 GPU 内存（消外部碎片、支持灵活调度），但**粒度是 token 粒度**，无法表达"同一请求内 token 的驻留状态不同"。eLLM 把该抽象扩展为 **token-layer 粒度**：每个物理单元存"一个 token block × 一个 F 层组"的注意力状态，携带标识其层范围的元数据；一个逻辑到物理的映射表跟踪这些**非连续**的 token-layer 单元。

### 3.2 数据结构
- **重塑**：原 token-block 视图（token × 全层）重塑为 token-layer block table（token block × 层组）。
- **映射表字段**（每个条目）：
  ```
  seq_id | token_id | layer_id | logical_block_id | physical_block_id | #filled
  ```
  #filled = block 内有效 token 数（尾 block 部分填充）。
- **层分组**：层维按 F 层一组；F 小 → 控制更细、临时内存浪费更少；F 大 → 元数据/查找开销更小。默认 F=4。
- **host/GPU 同布局**：host 驻留与 GPU 驻留状态用同一重塑布局，传输**无需布局转换**，简化按层物化。

### 3.3 F 敏感性（§6.5, Llama2-70B 实测）
| F | 内存碎片率 | map 开销占请求处理时间 | 吞吐 |
|---|---|---|---|
| 2 | 1.18% | 6.67% | 基准 |
| 4 | 1.28% | 3.84% | 最高（比 F=2 高 1.14×） |
| 8 | 2.30% | 1.47% | 比 F=4 低（碎片+临时浪费） |

### 3.4 内存稳定性
固定大小 token-layer block + PagedAttention 的 block 复用原则：请求结束后释放的 block 回**全局 free list**，无需压缩/GC；残余内部碎片主要来自部分填充的尾 block；论文 2.92 小时长跑无 free pool 耗尽。

### 3.5 vLLM 实现要点
- 扩展 block 元数据：加 token-layer 索引（seq/token/layer → logical → physical）。
- Attention 后端能按 (token block, layer group) 定位张量地址，只物化当前 decode 需要的状态。
- eviction/restore/重算都通过映射表读写，物理位置非连续不感知。

---

## M4. 临时工作区记账（§4.3.2, Eq.15）

### 4.1 机制原理
融合与重叠需要**临时 GPU 内存**来暂存物化出来的状态（重算的 KV、恢复的 KV、解码中间状态）。这些临时 buffer **会挤占可用于持久状态和请求准入的 HBM**——这是论文明确指出的"fusion 需要临时存储、可能降低 prefill 准入容量、恶化 TTFT"的权衡，所以必须进内存约束，并反馈给请求级控制器。

### 4.2 数学模型（Eq.15）

```
T_attn(π) = Σ_{l=1}^{L} ( a_l(S_swap) + b_l(S_rec) + c_l(S_dec) )
a(S) = Σ_{l∈S} ρ_attn     # 按"角色"计每层每 token 状态字节

三个独立 buffer：恢复用、重算用、解码用；
生命周期重叠时也按角色分别计数（即使层集合重叠）。
```

### 4.3 两阶段记账（反馈闭环）
1. **请求级**先用**保守上界**：按最大支持的恢复/重算层数（|S_swap|≤3、|S_rec|≤2）估算 T_attn。
2. **层级优化器**选定具体重叠/融合配置后，用 Eq.15 **精化** T_attn，连同线程配置反馈给下一轮请求级迭代。

### 4.4 与你的建模对应
你建模里的临时 buffer 就是：重算的 KV(L, prefix)（u×ρ_attn 字节）+ 一层在途的 prefix hidden states（u×hidden×dtype 字节）+（若加恢复路径）恢复的 KV buffer。这些在"用后即释放"之前都占 HBM，要计入容量规划（论文把它和持久驻留分开算，你实现时也要分开记账）。

---

## M5. 融合内核与线程模型（§4.3.2, Fig.10, Eq.16, §5）

### 5.1 机制原理
- **垂直融合**：重算路径内部（如 QKV 投影、RoPE、attention、dense 等重算子算子）融合成一个 K1 内核，减少 launch/中间状态访问。
- **水平融合**：K1（重算）与 K2（解码）合入**同一个内核**，两个工作分区并发执行。动机：K1 是 prefill-like、compute-bound；K2 消费历史状态、memory-bound；分开跑会各自让带宽或 SM 空闲（Observation 3）。
- 融合核（Stream B）与 host→GPU 恢复（Stream A）再并发，构成双层重叠。

### 5.2 工作负载分配（Eq.16）

```
W1 = Σ_{l ∈ S_rec} v_rec,state(l)     # K1: 重算状态工作量
W2 = Σ_{l ∈ S_dec} v_dec,attn(l)      # K2: 解码 attention 工作量
```

注意：**融合路径只含这两项**；非融合的 dense、MoE、collective 仍在 t_GPU 里（Eq.5），不进融合核；vocab projection（logits）也不在融合路径（只物化中间状态）。

### 5.3 线程分配（具体设计）
```
thr      = 每 block 总线程数
t        = 分配给 K1 的线程数
thr − t  = 分配给 K2 的线程数

约束：
  thr ≤ 1024            # A100/H100 每 block 线程上限
  t 与 thr−t 均为 32 的倍数   # warp 对齐（论文引 CUDA Programming Model [35]）
```

- **配置映射**：把 W1/W2 比值映射到**预编译**的线程配置变体（编译成 .so 共享库，论文 1700 行 CUDA 编译出多套变体）；运行时按当前估算的重算/解码工作量选变体。
- 论文 Fig.5b 评估了 K1:K2 = 0.5:0.5 与 0.75:0.25 等比值下的吞吐/TPOT/TTFT 收益。
- 线程配置选择开销 ≈ 0.03ms（实测），很小。

### 5.4 与你的建模对应
- 你的 kernel_a + kernel_b = 论文的 K1 + K2；"解码和重算同时处理"= 单内核内 warp 划分并发 + 与 swap stream 的重叠，**不是两个独立内核**。
- 你的 kernel_b 把"前缀 token 的整层前向（attention+dense）"也算进去了——这在论文里属于 **K1（重算路径）** 的工作（重算要再生 hidden 状态），K2 只覆盖新 token 的解码 attention。做 workload 估算（W1/W2）时别分错边。
- 融合方式 1（同层）对应论文 K1∥K2 的标准形态；方式 2（提前一层）对应 Fig.5a 的 "future layer 状态与当前层解码并发重算"，注意依赖：kernel_a(L+1) 需要 h_L(prefix)（由 kernel_b(L) 产出），实际是跨层迭代的细粒度流水。

### 5.5 工作示例（粗估）
W1/W2 估算：W1 ∝ u×L（重算 u 个 token 的前向），W2 ∝ 1×L（解码 1 个新 token）。
- u=32（AGENT.md 示例）：W1:W2 ≈ 32:1 → 选 t 大的变体（如 thr=1024, t=992, thr−t=32）。
- u=2：W1:W2 ≈ 2:1 → 选 t≈683→取 672（32 倍数），thr−t=352。
- 预编译几个档位（如 32:1、8:1、2:1、1:1），运行时按 W1/W2 就近选。

---

## M6. 延迟模型（roofline，§4.2.1, Eq.5–9, 13）

### 6.1 为什么不能只用 FLOPs
decode 延迟还受 HBM 流量、host→GPU 传输、路由不均、collective、同步影响；且**不是所有资源全局重叠**，要按层依赖组合。所以论文把模型分为三层输入：

| 类别 | 内容 |
|---|---|
| 模型参数 | 层数 L、hidden、注意力头布局、ρ_attn（Eq.2）、词表、MoE 配置 |
| 硬件常量（需 profiling） | 有效算力吞吐、HBM 带宽、host→GPU 带宽、copy-engine 并发、launch 开销、sync 开销 |
| 运行时状态 | 序列长度、队列长度 q、可用 HBM、batch 组成、专家驻留 x、路由统计 |

### 6.2 组合公式（Eq.5）
```
T_iter = Σ_l max( t_GPU(l), t_swap(l) ) + t_launch + t_sync
```
- Σ 保留层间串行依赖；max 只作用于并发的 swap/计算。
- 每 kernel phase 用 roofline：t = max(FLOPs/π_eff, bytes/BW_eff)。
- 队列时间不计入 TPOT（属 TTFT）。

### 6.3 精度验证（论文 §6.6）
- V100 上 Llama2-13B 延迟估计 MAPE = 13%。
- A100 上把有效算力/带宽常量扰动 ±20%：吞吐损失 ≤1.9%、TTFT 变化 ≤5.4% —— 对 profiling 误差鲁棒，但**每个新模型×GPU 组合仍需先 profiling**。

### 6.4 对你的实验
CUDA 13.1 环境需要重测的常量：HBM 带宽、PCIe 4.0×16 有效 H2D 带宽、launch/sync 开销、以及 Llama-13B 各算子的有效吞吐（vLLM profiler 或 nsys 实测）。

---

## M7. 自适应行为与边界条件（§6.9, §6.4, §6.7）

### 7.1 ρ 怎么变（自适应方向）
| 信号 | ρ 的调整 |
|---|---|
| HBM 压力大 / 队列长 | ρ↑（更多非驻留 → 更多并发） |
| compute 或 TPOT slack 紧张 | ρ↓（少重算；重算比恢复贵时转向恢复优先） |
| host→GPU 互联/copy engine 饱和 | ρ↓（少 swap；恢复贵的层切重算或驻留） |
| MoE 路由/专家执行主导关键路径 | ρ↓（更保守） |
| HBM 有余量 | ρ→0（全驻留，即你建模里 "keep all kv caches" 的情形） |

### 7.2 论文实测的驻留比（参考）
| 模型/工作负载 | 平均驻留比 | 持久注意力状态 HBM 节省 |
|---|---|---|
| Llama2-70B, ShareGPT | 0.64 | 36% |
| Llama2-13B/70B, L-Eval | 0.53 | ~47%（吞吐最高 3.1×） |
| Qwen2.5-32B（GQA）, ShareGPT | 0.56 | 44% |
| DeepSeek-V2-Lite（MLA/MoE）, L-Eval | 0.71 | 29% |

（MLA 足迹小 + MoE 开销大 → eLLM 选择更保守的驻留比。）

### 7.3 边界条件（论文明确"不该用"的场景）
- 全驻留已经放得下（HBM 不是瓶颈）；
- decode 已饱和 GPU compute；
- TPOT SLO 无余量做精确重算；
- host→GPU 恢复已饱和 copy engine 或互联。
这些场景下控制器应选低 ρ（保守），eLLM 无增益——**做实验时先确认你的负载落在"内存受限"区间**。

### 7.4 对你的实验
不要固定 drop_len：实现一个最小闭环——每迭代估 T_iter 与内存，若 T_iter > TPOT 则降 ρ/增 swap；若内存有余则升 ρ（或升 B）；"HBM 有余量 → 全保留"只是 ρ=0 的端点。

---

## 8. 七个机制的关系（闭环总图）

```
                    ┌─────────────────────────────────────────────┐
                    │  请求级控制器 (M2)                           │
                    │  max B/T_iter                               │
                    │  s.t. 内存 ≤ M_GPU,  T_iter ≤ SLO_TPOT       │
                    │  SLSQP 连续松弛 → 整数邻域 → FCFS 准入        │
                    └───────────────┬─────────────────────────────┘
                                    │ (B, ρ) → u = ρ·s
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │  层级优化器 (M1)                             │
                    │  Eq.14: min max(t_swap(S_swap), t_rec(S_rec))│
                    │  |S_swap|≤3, |S_rec|≤2 → S_swap/S_rec 划分   │
                    └───────────────┬─────────────────────────────┘
                                    │ 物化方案 + 融合配置 (M5)
                                    ▼
                    ┌─────────────────────────────────────────────┐
                    │  执行层 (M3 内存管理器 / M5 融合核)           │
                    │  Stream A: host→GPU 恢复 (M1)                │
                    │  Stream B: 融合核 K1∥K2, 线程 t : thr−t       │
                    │  临时工作区 T_attn (M4)，用后即释放            │
                    └───────────────┬─────────────────────────────┘
                                    │ 反馈: T_attn 精化值、线程配置、实测延迟
                                    ▼
                          回到请求级下一轮迭代（闭环）
```

## 9. 速查：你实现时最可能漏掉的点
1. 非驻留状态的**备份**：丢之前先拷 host，否则没有 swap 可走（M1）。
2. **u 是动态的**：u = ρ·s，ρ 每迭代可调；不是固定 drop_len（M2/M7）。
3. **融合核 = 单内核内 warp 划分**（K1 与 K2 同核并发），不是两个 kernel 轮发（M5）。
4. **临时 buffer 计入内存约束**：T_attn 与 R_attn 分开记账（M4）。
5. **logits 在融合路径外**：融合核只物化中间状态，最后一层后由 LM head 出概率（0.1）。
6. **h_L(prefix) 的在途生命周期**：只保留一层，kernel_a(L+1) 消费后再丢（0.1）。
7. **环境常量重测**：CUDA 13.1 下 HBM/PCIe 带宽、launch/sync 开销都要 profiling（M6）。
