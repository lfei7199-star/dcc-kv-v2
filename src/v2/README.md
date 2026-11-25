# src/v2 — CPU reference（2A 链 / 2C shared-mode / 2D destination-conditioned）

**这一段代码不产出任何论文结果。** 它只做三件事：把归并数学写对、把结构验收逻辑写死、把每条边的构造记录连同口径一起落盘。

- 后端：纯 Python + 标准库，`float64`。**没有任何 CUDA、没有 torch、没有 numpy**
  （本机装不上：PyPI 与清华镜像均返回 `No matching distribution`）。
- 判据：`tests/` 下 350 个 unittest，全部在 CPU 上跑，`OK`（108 + 88 记录规范 + 59 Q8 + 95 pilot）。
- 权威边界：本目录代码**不构成 2A 通过**。`accept.evaluate()` 在 CPU 上最高只能返回
  `BLOCKED_CPU_NO_GPU_EVIDENCE`；`two_d.evaluate()` 最高只能返回 `PASS_CPU_SEMANTICS`。
  两者都不会返回裸 `PASS`——这是刻意的。

## 文件

| 文件 | 职责 | 对应冻结判据 |
|---|---|---|
| `cpu_ref/state.py` | `(m, l, acc)` 状态与 online-softmax 归并；`merge` 可结合可交换 | 2D 顺序无关的基础 |
| `cpu_ref/sp_attention.py` | 按 `world_size` 切分、逐 rank 本地 + 远端块计算、resident/staging 分账 | 2A-01 / 2A-02 / 2A-03 |
| `cpu_ref/accept.py` | 结构验收器：分片比、负向臂拒绝、远程块数、Q8 数值门（含 NaN/Inf、shape、边界覆盖、reference 身份）、backend | 2A-01 … 2A-05 + 04b/04c |
| `cpu_ref/q8.py` | **Q8 冻结值**：`relative_L2 ≤ 1e-5`、三态 tolerance 来源、预登记位置规则；**只 import math** | 2A 数值门策略 |
| `cpu_ref/compact.py` | **AM-HighestAttnKeys 构造链**：softmax-RMS 选 key → 盒约束 β → OLS Value；`build_shared`(B′) vs `build_per_destination`(C) | 2C（B′）vs 2D（C）的归因 |
| `cpu_ref/record.py` | **按 Source→Destination 边的构造记录**：口径常量 + schema + 校验 + 载荷核算 | 2C 记录模板；2D-05 |
| `cpu_ref/two_d.py` | 1 源 × 3 目的端拓扑、六项检查、序列化往返、记录接口检查 | 2D-01 … 2D-08 |
| `cpu_ref/envelope.py` | **统一记录外壳**：schema 门禁、三态语义、需求矩阵、跨边汇总检测、判决折算；**不 import compact/record** | 2A / 2C / 2D 共用记录规范 |
| `cpu_ref/adapter_2a.py` | 2A → 记录外壳的翻译器（`accept` 的状态映射 + CPU 天花板钳制） | 2A 记录；保证 2A 不依赖求解器 |
| `cpu_ref/adapter_compact.py` | 2C / 2D → 记录外壳的投影（逐边五块诊断，不重算） | 2C 记录；2D-07 / 2D-08 |

## 构造链（2026-11-25 起对齐 AM 原文）

| 环节 | 规定 | 出处 |
|---|---|---|
| Key Selection | `a_i = softmax(q_i Kᵀ)`，`s_j = RMS_i(a_ij)` | AM §3.3 |
| β | `w = argmin_{e⁻³≤w≤e³} ‖Aw−m‖²`，`β = log w`，warm start 为无约束 LS 后裁剪，再 2 步投影梯度 | AM §3.2 + C.2 |
| Value | OLS，**无正则** | AM §3.2 / C.2 |
| 逻辑长度 | 保留 `logical_len = T`（**必须由调用方传入源的 T，不是分片自己的长度**） | AM §2 |

**硬前置**：`n_ref ≥ 2t`（U3 裁定）。不满足或列秩不足 ⇒ `RankDeficientError` 带 stats 抛出，
**没有静默加 λ 的路径**。ridge 与标量 β 只作为显式消融臂（`value_mode="ridge"` /
`beta_mode="uniform"`）存在，不得挂 AM 的名字。

## 诊断口径（唯一来源：`record.DEFINITIONS`）

| 量 | 定义 | 陷阱 |
|---|---|---|
| `beta_bind_hi/lo` | `\|β_j ∓ 3\| ≤ 1e−6`，容差**冻结、无参数入口** | 容差若可传入，看完结果再挑就永远通过 |
| `beta_dead` | 贡献 `c_j = w_j·Σ_i A_ij` < `1e−6 × Σ_j c_j` | 盒内 `w_j ≥ e⁻³ > 0`，**不是** `w_j=0`，**不是** `bind_lo` |
| `mass_residual` | `‖Aw−m‖₂/‖m‖₂`，全部 calibration Query **堆叠后一次取范数** | 不是逐 Query 比值再平均；原始残差与分母同时保存 |
| `beta_design.*` | β 拟合矩阵 `A`（`n_ref × t`）的 rank/cond/σ | **与 value_design 不得共用** |
| `value_design.*` | Value OLS 设计矩阵 `X`（`n_ref × t`）的 rank/cond/σ | 合成实测两者可差 41–59 倍 |

## 刻意的设计约束

1. **tolerance 必须外部传入，且来源被限定。** `check_numeric(..., tolerance=None)` 直接抛异常；
   Q8 冻结为 `relative_L2 ≤ 1e-5`（`source=preregistered`），**measured 运行只接受这一个来源**，
   `1e-9` 只能标 `test_value`。把 1e-9 标成 `preregistered` 会被 `q8.validate_policy` 拒绝。
2. **负向臂由同一验收器拒绝。** `sp_forward(..., all_gather=True)` 产出的全量复制臂，
   必须被 `evaluate()` 判 `2A-01`/`2A-02` FAIL，而不是人工标 FAIL。
3. **resident 与 staging 不合并。** `resident_kv_bytes` 只算自有 span；
   远端块计入 `staged_kv_bytes`。两者相加会掩盖真实驻留。
4. **CPU 不可伪装 GPU。** CPU backend 上 `2A-05` 为 `NOT_APPLICABLE_NO_CUDA`，折出 `BLOCKED`；
   合成 workload 不能标 `measured_hardware`，CUDA backend 不能降级成 `structural_cpu`。
5. **诊断不可省略。** 每个 compact 都带 `diagnostics.beta` / `diagnostics.value`，
   `record.validate()` 会拒绝缺块或口径漂移的记录。
6. **求解与诊断分离。** `solver.method`（答案怎么来）与 `*_design.method`（rank/cond 怎么来）
   是两个字段。CUDA 的 `gels` 假定满秩且不给出可用 rank，所以 GPU 运行必须单独诊断一遍。
   `solver_policy()["fallback"] == ""`：失败就记录失败，**禁止自动切 ridge/pinv**。
7. **NaN 先于 L2。** `check_numeric` 先扫非有限值，命中即返回且 `rel_l2` 保持 `None`——
   不会出现"相对误差很小，但张量里有 Inf，于是通过"。shape 不等直接 FAIL，不用 `zip` 截断后硬算。
8. **Q8 是三行。** 2A-04（`rel_l2 ≤ 1e-5`）/ 2A-04b（预登记位置全覆盖且有限）/ 2A-04c（reference SHA 一致）。
   分开是因为失败含义不同：reference 找不到是**比较无效**，不是"误差很大"。
9. **记录按边，不按实验。** `payload.counted_for` 恒为 `"this edge only"`；不存在跨边总字节字段。
   B′ 是 1 次构造服务 3 个目的端，C 是 3 次构造——这个不对称本身要被记录，不能被汇总抹平。
10. **共用外壳、分离诊断。** 2A 与 2D 共用一个 `experiment_record` 外壳，但压缩诊断块在 2A 上
   **结构性地不存在**（`NOT_APPLICABLE`），理由写在记录里。`NOT_APPLICABLE`（阶段没有这个量）
   与 `MISSING`（阶段该有、这次没有）是两个状态；前者是性质，后者是证据缺口。
11. **记录不能自我批准。** `verdict` 由 `fold_verdict(checks)` 重算后与声明值比对，手改的
   `"PASS"` 会被拒；`structural_cpu` 证据级的天花板是 `PASS_CPU_SEMANTICS`，够不到 `PASS`。

## 记录规范（2026-11-25 整合，详见 `design/16`）

| 检查范围 | 2A | 2C | 2D |
|---|---|---|---|
| Rank / topology | 必须 | 证据级 | 证据级 |
| Owner KV resident bytes | 必须 | 证据级 | 证据级 |
| Remote staging | 必须 | 证据级 | 证据级 |
| 全量 KV 复制负向测试 | 必须 | 证据级 | 证据级 |
| 实际通信记录 | 证据级 | 证据级 | 证据级 |
| β 盒约束诊断 | **不适用** | 必须 | 必须 |
| Beta / Value rank、cond | **不适用** | 必须 | 必须 |
| Mass residual | **不适用** | 必须 | 必须 |
| Destination-specific 语义 | **不适用** | **不适用** | 必须 |

「证据级」（`required_evidence`）= 缺了不判记录失效，判 `BLOCKED_CPU_NO_GPU_EVIDENCE`；
「必须」（`required`）= 缺了直接拒收。矩阵是数据（`envelope.REQUIREMENT_TABLE`），
校验器由它驱动，测试反过来断言矩阵本身。

## 已知未实现（不要在论文里当作已完成）

- `manual_decoder_forward`
- 84 轮 NCCL 接线、真实 `all_to_all_v` 传输
- CUDA backend / `sdp` kernel 的任何实测
- **真实规模**：本实现的 QR 是 `O(n_ref·t²)`。实测 t=16…256，局部指数 1.24→2.72，
  外推 t=1637 约 **23–38 min/edge**（`results/cpu/cpu_scale_timing.json`）。
  也就是说 t=1637 **不是算不动，是不划算**；真正的阻塞是没有真实 KV 张量。
  真实运行必须走 `torch.linalg.lstsq`。
- 逐 KV head / 逐层的批量化：接口按 `(layer, kv_head)` 语义设计，CPU 参考只跑单 head。
- `logical_len` 的 RoPE 相位端到端验证（2D-06b = `BLOCKED_CPU_NO_GPU_EVIDENCE`；随 2A GPU 一起做）
- 真实数据、真实模型、真实长度（130944）

## 跑法

```bash
python -m unittest discover -s tests -p "test_*.py"    # 350 tests
python tools/measure_2c_specialization.py              # -> results/cpu/2c_specialization.json
python tools/run_2d_closure.py                         # -> results/cpu/2d_closure.json
python tools/diag_am_fidelity.py                       # -> results/cpu/diag_am_fidelity.json
python tools/time_cpu_scale.py                         # -> results/cpu/cpu_scale_timing.json
python tools/validate_records.py                       # 从文件回读三份 stage 记录并校验
```

## 合成读数（advisory，不可引用为论文结果）

`status = advisory_only_synthetic`。所有 JSON 都不进 Table 4。

**2C 专项（每档 n_ref=2t）**：45 格中 C 优于 B′ 42 格。

| budget | n_ref/边 | mean B′ | mean C | C 更优 | bind_hi(C) | dead | beta_cond / value_cond |
|---|---|---|---|---|---|---|---|
| 4 | 8 | 0.5446 | 0.3833 | 7/9 | 0.64 | 0.00 | 48 / 55 |
| 6 | 12 | 0.3905 | 0.2433 | 8/9 | 0.30 | 0.00 | 813 / 4 677 |
| 8 | 16 | 0.4485 | 0.1407 | 9/9 | 0.11 | 0.00 | 2 640 / 156 981 |
| 12 | 24 | 0.2971 | 0.0398 | 9/9 | 0.03 | 0.00 | 50 991 / 153 431 |
| 24 | 48 | 0.0623 | 0.0040 | 9/9 | 0.00 | 0.00 | 1 938 621 / 79 442 301 |

**读法（必须连同混杂一起读）**：这一版里 budget 与 n_ref 同步变大，所以"大预算档 C 更优"
不能归因给预算。上一版（旧链、固定 n_ref=4）看到的「优势随预算放宽而消失」在新链下不再出现，
说明那条趋势至少部分是旧链的伪影。⇒ 四档 sweep 必须共用 `n_ref=13096`，否则这个混杂无法拆开。

**`dead` 恒为 0 是一个 null，不要包装**：盒内 `w_j ≥ e⁻³`，而 iid 高斯键的列和量级相当，
没有 key 会掉到 1e−6 以下。计数逻辑已用合成贡献向量单测（`test_count_dead_boundary_is_strict`），
但**该指标在当前数据上没有分辨力**，要等真实 KV 上"几乎不被任何 calibration Query 关注的 key"。

**条件数不能外推**：这组合成 Query 被限制在 2 维子空间，设计矩阵天然接近奇异。
表里两组 cond 的用途是证明"beta 与 value 必须分开记"，不是预测 Qwen 上的条件数。

**AM 保真度对照**：合成数据**不支持**「AM 更优」——40× 与 5× 上旧链反而更好。升级的正当性是
保真性（B′/C 若不是 AM 的忠实实现，与 AM 对照就是方法定义错误），不是合成性能。
详见 `design/14_AM对齐验收报告_2026-11-25.md` §8。

**2D 闭环**：`results/cpu/2d_closure.json`，构造语义 overall = `PASS_CPU_SEMANTICS`，
`blocked_requires_gpu = ['2D-06b', '2D-08']`。详见 `design/15`。

**同一文件里的两份 stage 记录**：`records["2A"]` 与 `records["2D"]` 是同一种对象、不同阶段的事实。
2A 记录有真实 dense SP 读数、压缩块 `NOT_APPLICABLE`；2D 记录有逐边五块诊断、SP 块 `MISSING`
（这次没有跑 SP，是缺口不是不适用）。两者判决都是 `BLOCKED_CPU_NO_GPU_EVIDENCE`，
但 `partial.compact` 在 2D 上是 `PASS_CPU_SEMANTICS`。**这是两个判决对象，不是两个口径**：
构造语义通过，阶段整体因硬件证据缺失而阻塞。详见 `design/16`。

**Q8 已冻结**：`relative_L2 ≤ 1e-5`（`source=preregistered`，禁止事后调整），见 `q8.py` 与 `design/17`。
2A 记录里 Q8 三行（`sp.numerical` / `sp.numerical_coverage` / `sp.reference_identity`）在 CPU 闭环上
均为 `PASS_CPU_SEMANTICS`，整体仍 `BLOCKED`——因为 `sp.backend` 与 `sp.communication` 没有硬件证据。
`1e-9` 仍标 `test_value`，不会因为 Q8 冻结而变成门。
