# DCC-KV v2 — CPU Reference Implementation

Destination-Conditioned Compact KV：面向超长文本协同推理的目的端条件化紧凑 KV 通信。

本仓库是该工作的 **v2 独立重写**，当前只包含 **CPU 参考实现、结构验收逻辑、以及 pilot 统计与校验器**。
它**不产出任何论文结果**，也**不代表 GPU 实验已经跑过**。

---

## Experimental status

> 以下四行是仓库当前的事实状态。任何把本仓库的 CPU 通过当作 GPU 实验结论的解读都是误读。

| 项目 | 状态 |
|---|---|
| `GPU submissions` | **0** |
| `real pilot data` | **0**（尚未采集：仓库内没有任何模型输出、评分或 `f1` 记录） |
| `Δ_DCC` | **NOT_TESTABLE**（非劣边界 `delta` 未由预登记提供，pilot 无权自行设定） |
| `819-token budget` | **not authorized**（未获授权，未执行） |

对应的代码级约束是硬的，不是注释：

- `accept.evaluate()` 在 CPU 上**最高只能返回** `BLOCKED_CPU_NO_GPU_EVIDENCE`；
- `two_d.evaluate()` 最高只能返回 `PASS_CPU_SEMANTICS`；
- 两者都**不会返回裸 `PASS`**；
- `results/cpu/*.json` 全部自标 `advisory_only_synthetic` / `timing_only` / `PASS_CPU_SEMANTICS`，**不进论文表格**。

---

## Project overview

长上下文协同推理中，把每个目的端各自需要的那部分 KV 压缩下来再传输，比搬运全量 KV 更省通信。
本工作的核心系统问题是：**变长紧凑 KV 的异步 All-to-Allv 传输，与顺序无关的 Online Softmax 归并**。

压缩机制本身（Key 选择 / 质量偏置 / Value 回归）沿用 Attention Matching 的构造，**不是本工作的原创**；
本仓库做的是把归并数学写对、把结构验收逻辑写死、把每条边的构造记录连同口径一起落盘。

v2 与早期实现是**两条独立工程线**：本仓库不承接、不回推任何旧提交历史，也不把旧实验记录混入。

## Current scope

**包含**

- 2A（真实 K/V 分片 + 层内 SP attention）的 CPU 语义参考与结构验收器
- 2C（Shared compact，B′）与 2D（Destination-conditioned，C）的构造链与逐边记录
- 统一记录外壳（schema 门禁、三态语义、需求矩阵、判决折算）
- Q8 数值门冻结值与其策略校验
- pilot 数据合同：G1–G17 门禁 + 允许的派生统计 + 聚类 bootstrap 方差估计
- 上述全部行为的单元测试（350 例）

**不包含（明确未做）**

- GPU 上的任何实测（无 CUDA kernel 读数、无 `sdp` backend 实测、无 NCCL 运行）
- 真实数据、真实模型、真实长度（130944）
- 真实 `all_to_all_v` 传输接线
- 手写 decoder forward（`manual_decoder_forward`）
- 论文表格中的任何数字

---

## Implemented components

`src/v2/cpu_ref/`（纯标准库，`float64`，无 torch / numpy / CUDA）：

| 文件 | 职责 |
|---|---|
| `state.py` | `(m, l, acc)` 状态与 online-softmax 归并，`merge` 可结合可交换 |
| `sp_attention.py` | 按 `world_size` 切分，逐 rank 本地 + 远端块计算，resident / staging 分账 |
| `accept.py` | 结构验收器：分片比、负向臂拒绝、远程块数、Q8 数值门、backend 一致性 |
| `q8.py` | Q8 冻结值 `relative_L2 ≤ 1e-5`、三态 tolerance 来源、预登记位置规则（只 import math） |
| `compact.py` | AM 构造链：softmax-RMS 选 key → 盒约束 β → OLS Value；`build_shared`(B′) vs `build_per_destination`(C) |
| `record.py` | 按 Source→Destination 边的构造记录：口径常量 + schema + 校验 + 载荷核算 |
| `two_d.py` | 1 源 × 3 目的端拓扑、六项检查、序列化往返、记录接口检查 |
| `envelope.py` | 统一记录外壳：schema 门禁、三态语义、需求矩阵、跨边汇总检测、判决折算 |
| `adapter_2a.py` | 2A → 记录外壳的状态映射（含 CPU 天花板钳制） |
| `adapter_compact.py` | 2C / 2D → 记录外壳的投影（逐边诊断，不重算） |
| `pilot.py` | pilot 数据合同：G1–G17 门禁、配对差、派生统计、聚类 bootstrap |

设计上的硬约束（每一条都有对应测试，不是文档承诺）：

1. tolerance 必须外部传入且来源受限；`1e-9` 只能标 `test_value`，标成 `preregistered` 会被拒。
2. 负向臂必须由**同一个验收器**拒绝，不允许人工标 FAIL。
3. `resident` 与 `staging` 字节**不合并**——相加会掩盖真实驻留。
4. CPU 不可伪装 GPU：CPU backend 上 backend 检查为 `NOT_APPLICABLE_NO_CUDA`，折出 `BLOCKED`。
5. NaN 先于 L2：先扫非有限值，命中即返回，不会出现「相对误差很小但有 Inf 因此通过」。
6. 记录不能自我批准：`verdict` 由检查结果重算后与声明值比对，手改的 `PASS` 会被拒。
7. 求解与诊断分离：`solver.method` 与 `*_design.method` 是两个字段，失败不许自动降级到 ridge。

---

## CPU validation

```bash
python -m unittest discover -s tests -p "test_*.py"
```

实测：**Ran 350 tests, OK**（约 8 s，纯 `unittest`，无 pytest 依赖，无 GPU）。

其中 1 例（`test_q8.TestBoundaryPositions.test_matches_the_execution_document`）需要回读本地冻结链里的
常量文件做交叉核对。该文件不在本仓库，因此在干净克隆中这一例**显式 skip**（349 pass + 1 skip），
不会被静默当作通过。

| 测试文件 | 用例 |
|---|---|
| `test_2a_sp.py` | 11 |
| `test_2c_compact.py` | 6 |
| `test_2d_conditioning.py` | 29 |
| `test_am_properties.py` | 31 |
| `test_diagnostics.py` | 31 |
| `test_envelope.py` | 88 |
| `test_q8.py` | 59 |
| `test_pilot.py` | 95 |

---

## Pilot validator

`tools/validate_pilot.py` 是 pilot 数据合同的 CLI 入口。它读行级 `jsonl` 与其元数据块，
先跑 G1–G17，**只有在没有任何门禁阻塞时**才计算设计允许的派生统计。

```bash
python tools/validate_pilot.py results/pilot                     # 目录内含 meta + rows
python tools/validate_pilot.py --rows r.jsonl --meta m.json
python tools/validate_pilot.py results/pilot --delta-pp 1.5 --out derived.json
python tools/validate_pilot.py results/pilot --json
```

退出码：`0` 全部门禁通过，`1` 有门禁失败或无数据，`2` 用法错误。

它**无论传什么参数都不会**：

- 自行发明或调整 `delta`（边界只来自 `--delta-pp`，即来自预登记，绝不来自记录本身）；
- 让 `NI_PASS` / `NI_REJECTED` 具备权威性（每条记录恒带 `verdict_authority = false`；单种子下 CI 天然不含种子方差）；
- 写入 registry / unlock / ledger / prereg，或提交任何任务。

---

## Statistical design

### Pilot input shape（统计摘要，不含输入内容）

> The pilot manifest contains 200 evaluation rows grouped into 182 input clusters
> (164 singleton clusters and 18 duplicate-input clusters).

这一评测结构决定了 `se_cluster_w` 所使用的簇大小权重 `m_c`
（*This evaluation structure determines the observed cluster-size weights
`m_c` used by `se_cluster_w`.*）

它是当前 pilot 数据的输入结构，不是算法本身的普适定义。逐行身份映射
（`clusters.json`、逐行 `input_sha256`）与 200 行评测集本身属于**实验冻结输入**，
不在本仓库——它们随 pilot 工作区管理，不随公开代码发布。

### 方差估计

方差估计是本仓库的方法论重点，已冻结：

```
SE_cluster_w  =  uniform-cluster bootstrap
              +  Hájek row-weighted replicate statistic

replicate:  Δ̂* = Σ_c k_c m_c ȳ_c / Σ_c k_c m_c      k_c ~ 均匀抽簇计数
target:     Δ  = Σ_c m_c ȳ_c / Σ_c m_c  =  (1/200) Σ_i d_i
```

为什么不用「抽簇后对各簇均值再平均」：200 行里 18 个簇各含两行、164 个簇各含一行。
等权平均会把目标参数**静默改成**簇均值平均，那 18 个双行簇被系统性降权——那是另一个参数。
比值形式把目标锁在 200 行均值上，并且对 `d_i` 的整体平移保持不变。

口径分层（只有一项进 CI）：

| 量 | 用途 |
|---|---|
| `se_cluster_w` | **唯一**用于区间估计的方差 |
| `se_iid` | 诊断 |
| `deff_empirical = (se_cluster_w / se_iid)²` | 诊断 |
| `deff_model = 1 + 0.18·rho_d` | 诊断 |
| `rho_D` / `rho_d` | 诊断 |

`deff_empirical` 与 `deff_model` 相差超过 0.10 时标 `model_misfit`，但**不改动** `se_cluster_w`——
用模型方差替换实测方差是缺陷，不是修正。`DEFF < 1` 只标记，不判失败。

---

## Reproducibility

无外部依赖，克隆即跑（Python ≥ 3.10，纯标准库）：

```bash
python -m unittest discover -s tests -p "test_*.py"     # 350 tests
python tools/validate_records.py                        # 从文件回读三份阶段记录并校验
python tools/measure_2c_specialization.py               # -> results/cpu/2c_specialization.json
python tools/run_2d_closure.py                          # -> results/cpu/2d_closure.json
python tools/diag_am_fidelity.py                        # -> results/cpu/diag_am_fidelity.json
python tools/time_cpu_scale.py                          # -> results/cpu/cpu_scale_timing.json
```

`results/cpu/` 下的四份 JSON 是**可重建的合成/计时产物**，不是实测数据。

---

## Repository layout

```
src/v2/cpu_ref/   CPU 参考实现（本仓库主体）
tools/            校验器与可重建的合成诊断脚本
tests/            350 例 unittest
results/cpu/      合成/计时产物（advisory_only，不进论文）
design/           设计冻结与审查记录（13–19、22–25）
```

## Design records

`design/13`–`19`、`22`–`25` 记录了从问题提出 → 设计 → 失败发现 → 统计纠错 → validator 落地 → 审查冻结的过程。

**部分设计原件不在此仓库。** 它们保存在本地冻结链中，因正文包含本地绝对路径等信息，当前不直接公开。
后续若确有公开需求，会另做 `docs/sanitized/` 副本，并在副本内明确标注「非冻结原件、原件 SHA256 保留」——
不会以修改原件的方式产出「伪原件」。

## Current limitations

- **没有真实数据。** 所有读数来自合成或小规模验证；`dead` 类指标在当前数据上没有分辨力。
- **条件数不可外推。** 合成 Query 被限制在低维子空间，设计矩阵天然接近奇异；表里的 cond 只用于证明
  「β 与 value 必须分开记」，不能预测真实模型上的条件数。
- **AM 保真度对照不支持「AM 更优」。** 升级的正当性是保真性，不是合成性能。
- **规模不经济。** 本实现的 QR 是 `O(n_ref·t²)`；实测 `t = 16…256` 局部指数 1.24 → 2.72，
  外推 `t = 1637` 约 23–38 min/edge。也就是说大预算不是算不动，是不划算；真实运行必须走
  `torch.linalg.lstsq`。
- **单 head。** 接口按 `(layer, kv_head)` 语义设计，CPU 参考只跑单 head，未做逐层/逐 head 批量化。
- **RoPE 相位端到端验证缺失**（`logical_len` 相关检查为 `BLOCKED_CPU_NO_GPU_EVIDENCE`，随 GPU 阶段一起做）。
- **无 GPU 证据。** 2A 的 backend 与通信两项没有硬件证据，因此整体判决恒为 `BLOCKED`，
  尽管构造语义部分为 `PASS_CPU_SEMANTICS`。

## Licensing

**尚未授予任何许可证。** 本仓库当前未附带开源许可，版权保留；在代码与文档的公开授权范围确认之前，
不假定任何使用、复制或再分发权利。许可证将在授权范围明确后正式补充。

## Independence

本仓库是 v2 独立工程，与早期实现的仓库历史**不共用**。两边的实验记录不互相覆盖：
本仓库的结论不得回填旧记录，旧记录的结论也不自动适用于本仓库。
