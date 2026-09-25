# 评测

## 当前门禁：retrieval_v2（离线记忆检索）

`bench/data/retrieval_v2.json`，三个虚构的王者荣耀玩家：

| 玩家 | 画像 | 记忆（已被取代） | 查询 | 用途 |
|---|---|---|---|---|
| `p_archer` | 射手，大学生（即 v1 的玩家） | 42（3） | 70，其中负例 20 | **dev**，只在这里调参 |
| `p_mage` | 中路法师，程序员 | 37（3） | 52，其中负例 15 | **test** |
| `p_jungle` | 打野，高中生 | 34（3） | 50，其中负例 15 | **test** |

test 集的两位玩家在提交 `e2d9bce` 中封存，那时还没有任何系统在上面跑过。候选系统在 `3bbf901` 中预登记，之后才跑了一次 test。

**查询类型**：
- direct：查询和记忆有字面重合
- paraphrase：换了一种说法（"开不开麦" ↔ "语音"）
- update：必须返回最新值（段位、手机、昵称的变化）
- multi：需要同时找到 2 条记忆
- negative：和玩家无关，应当什么都不返回

**指标**（k=3，和聊天机器人每轮注入的记忆条数一致）
- `MemScore@3`（主指标）：对所有查询取平均。有 gold 的查询计 recall@3；负例返回空计 1，否则计 0。
- `Recall@3`、`MRR`：只在有 gold 的查询上计算。
- `Abstain`：负例中正确返回空的比例。

复现：`python -m bench.run_retrieval --split test --systems v0-keyword,v1-hybrid,v2-hybrid`

### test 结果（102 条查询）

| 系统 | MemScore@3 | Recall@3 | MRR | Abstain | paraphrase | ms/查询 |
|---|---|---|---|---|---|---|
| v0-keyword（原 main） | 0.750 | 0.715 | 0.678 | 0.833 | 0.379 | 8 + 1 次 LLM 调用 |
| v1-lexical（jieba + BM25） | 0.735 | 0.667 | 0.600 | **0.900** | 0.276 | 0.3 |
| v1-hybrid（+ bge-small-zh） | 0.843 | 0.875 | 0.778 | 0.767 | 0.724 | 7 |
| **v2-hybrid（+ jina-v2-base-zh）** | **0.892** | **0.903** | 0.806 | 0.867 | **0.793** | 37 |
| v2-hybrid+rerank（bge-reranker） | 0.877 | 0.882 | **0.850** | 0.867 | 0.759 | 103 |

与 v2-hybrid 的配对 bootstrap（10k 次，按查询重采样）：

| 对比 | ΔMemScore | 95% CI | 胜 / 负 |
|---|---|---|---|
| v2-hybrid − v0-keyword | +0.142 | [+0.049, +0.235] | 20 / 5 |
| v2-hybrid − v1-hybrid | +0.049 | [−0.010, +0.118] | 8 / 3 |
| v2-hybrid − v2-hybrid+rerank | +0.015 | [−0.049, +0.078] | 7 / 5 |

**结论**：v2-hybrid 在所有指标上都超过 v0，且显著。门禁判定 PROMOTABLE，已合入 main。相对 v1-hybrid 和 rerank 版本还不显著，所以更轻的 v1-hybrid 是可以接受的替代方案。

**v2-hybrid 仍然会错的地方**：
- 负例误召回："你是机器人吗""王者荣耀是哪家公司做的""你喜欢什么颜色"
- 抽象问法："我打游戏说话多吗" ↔ "经常开麦指挥"，"我喜欢反野吗" ↔ "入侵对方野区"
- 需要总结多条记忆的问题："我打野有什么特点"

### dev 调参记录（p_archer，70 条）

| 实验 | 最好的 MemScore@3 | 备注 |
|---|---|---|
| v0-keyword | 0.814 | 基线 |
| v1-hybrid（bge-small-zh，P0 阈值） | 0.871 | Abstain 0.75 |
| bge-small-zh + bge-reranker（阈值 −1，pool 5） | 0.871 | 慢 12 倍，multi 类下降 |
| bge-small-zh + jina-reranker | 0.393 | 正负分数大面积重叠，放弃 |
| bge-m3（Ollama） | 0.914 | Abstain 最多 0.85；需要额外起服务 |
| **jina-v2-base-zh** | **0.929** | `min_dense` 0.22–0.28、`margin` 0.15 这一段结果完全相同；取中间值 0.25 |

各向量模型的余弦尺度差别很大（负例最高分：bge-small 0.5、jina 0.37、bge-m3 0.63），所以阈值按模型分别校准，见 `RetrievalConfig.for_embedder` 和 `CALIBRATED_DENSE`。

### 如实说明

- v0 用 LLM 抽关键词的那一步，离线评测用 jieba TF-IDF 代替，v0 的分数可能被低估。
- 所有数据由维护者手写，存在作者偏差。test 虽然封存了，但写数据和做系统的是同一个人。
- 每个 test 玩家只有 50 条左右的查询，置信区间较宽。

## 历史：retrieval_v1

单个玩家、60 条查询，已被 v2 取代，结果保存在 `bench/results/retrieval_v1_test.json`。在 v1 test 上，v1-hybrid 的 MemScore 为 0.850，v0 为 0.800，但拒答率从 1.0 掉到 0.5，当时没有通过门禁。v2 的做法就是从这次教训来的：先封存 test、预登记候选、按模型校准阈值。

## 排行榜（main 的 SOTA 历史）

| 日期 | 系统 | 数据 | MemScore@3 (test) | ref |
|---|---|---|---|---|
| 2026-09-24 | v0-keyword | retrieval_v2 | 0.750 | `archive/v0-keyword-baseline` |
| 2026-09-24 | **v2-hybrid** | retrieval_v2 | **0.892** | `research/personal-memory-foundation` |
| 2026-09-24 | **P1**（检索持平 0.892；e2e_v2 test 0.448 → 0.759） | retrieval_v2 + e2e_v2 | 0.892 | `research/personal-memory-write-path` |
| 2026-09-24 | **P2**（检索 0.912；e2e_v4 test 0.693 → 0.765） | retrieval_v2 + e2e_v4 | 0.912 | `research/personal-memory-consolidation` |
| 2026-09-25 | **P3**（检索持平 0.912；e2e_v5 test 0.794 → 0.825） | retrieval_v2 + e2e_v5 | 0.912 | `research/personal-memory-facts` |
| 2026-09-25 | **P4**（检索 0.922；e2e_v6 test 0.482 → 0.563） | retrieval_v2 + e2e_v6 | 0.922 | `research/personal-memory-extraction` |
| 2026-09-25 | **P5b**（检索持平 0.922；e2e_v8 test 0.451 → 0.586） | retrieval_v2 + e2e_v8 | 0.922 | `research/personal-memory-subjects` |

## 端到端评测：e2e（写入 + 检索）

`bench/run_e2e.py`：用真实 LLM（本地 Ollama，默认 `qwen2.5:3b`）按日期依次写入多段对话，再在固定日期提问。检索到的前 3 条记忆（带事件日期、记录日期和"已过时"标记）作为证据，用确定性规则判分，不用 LLM 当裁判：

- fact / update / temporal / episodic / promise：证据中包含任一 `answer_any` 即算答对；ISO 日期也会展开成"M月D日"再匹配。
- update：如果证据里还出现过时的值（`stale_any`），也算错。
- trajectory：`answer_all` 里的每个状态都要出现。
- negative（从没提过的事）：必须什么都不返回。

主指标 `E2EScore` 是所有题目成功率的平均；另外报告 `Answer@3`、`StaleRate`（更新题里返回过时值的比例）和 `Abstain`。每个系统都跑 seed 0、1、2，结果合并。复现：`python -m bench.run_e2e --split test --seeds 0,1,2`；如果只改了判分规则，可以用 `--rejudge <结果文件>` 离线重算。

| 数据集 | dev | test（封存提交） | 题型 |
|---|---|---|---|
| `e2e_v1` | p_support、p_mage2 | p_tank、p_marksman2（`c58c35a`） | fact / update / temporal / negative |
| `e2e_v2` | e_marks、e_jungle | e_mid、e_support（`eb8add4`） | 以上四类 + episodic / trajectory / promise |
| `e2e_v3` | v3_archer、v3_tank | v3_mage、v3_jungle、v3_support（`b15eccc`） | 轨迹、更新、共享谓词负例加重 |
| `e2e_v4` | v4_mid、v4_adc | v4_jg、v4_sup、v4_top（`995e09d`） | 小段位变化、多属性开场白 |
| `e2e_v5` | v5_d1、v5_d2 | v5_t1、v5_t2、v5_t3（`ddee1cd`） | 事实题为主（类别问法、只在 episode 里的事） |
| `e2e_v6` | v6_d1、v6_d2 | v6_t1 … v6_t8，112 题（`c259842`） | 由不看代码的 agent 编写：一句多事实、亲属干扰、部分复述、近似负例 |
| `e2e_v7` | v7_d1、v7_d2 | v7_t1 … v7_t8，128 题（`e212511`） | 由不看代码的 agent 编写：事实是关于谁的（亲属事实、亲属变化、关于人的近似负例） |
| `e2e_v8` | v8_d1、v8_d2 | v8_t1 … v8_t8，128 题（`b9cd79b`） | 由不看代码的 agent 编写：现状更新（24 题，其中 8 题是亲属的变化）和关于人的题并重 |
| `e2e_v9` | v9_d1、v9_d2 | v9_t1 … v9_t8，128 题（`33fca06`） | 由不看代码的 agent 编写：各种说法的承诺（24）、3–5 步的轨迹（16）、从没答应过的负例（8） |

### e2e 记录

| 日期 | 系统 | 数据 | E2EScore | Abstain | StaleRate | 结论 |
|---|---|---|---|---|---|---|
| 2026-09-24 | v0-pipeline | e2e_v1 test | 0.219 | 1.000 | 0.000 | 小模型下几乎写不进记忆 |
| 2026-09-24 | main v2 | e2e_v1 test | 0.583 | 0.778 | 0.259 | e2e 基线（`bench/baseline_e2e.json`） |
| 2026-09-24 | v2 + 只读玩家 + 历史召回 | e2e_v1 test | 0.594 | 0.667 | 0.148 | **未通过**（+0.010 < +0.03，拒答率下降 0.11） |
| 2026-09-24 | main v2 | e2e_v2 test | 0.448 | 0.667 | 0.222 | e2e_v2 基线（`bench/baseline_e2e.json`） |
| 2026-09-24 | **P1** | e2e_v2 test | **0.759** | 0.667 | **0.111** | **通过**：+0.310，95% CI [+0.138, +0.483]，合入 main（`bench/sota_e2e.json`） |
| 2026-09-24 | main P1 | e2e_v3 test | 0.640 | 0.867 | 0.194 | e2e_v3 基线（`bench/baseline_e2e.json`） |
| 2026-09-24 | 巩固 + 偏好门槛 + 接地检查 | e2e_v3 test | 0.667 | 1.000 | 0.139 | **未通过**（+0.026 < +0.03；轨迹 0.278 → 0.056） |
| 2026-09-24 | main P1 | e2e_v4 test | 0.693 | 0.956 | 0.100 | e2e_v4 基线（`bench/baseline_e2e.json`） |
| 2026-09-24 | **P2**（按属性回忆 + 偏好门槛 + 接地检查） | e2e_v4 test | **0.765** | 0.933 | **0.000** | **通过**：+0.072（CI [−0.007, +0.157]，边缘显著）；轨迹 0.333 → 1.000；合入 main |
| 2026-09-25 | main P2 | e2e_v5 test | 0.794 | 1.000 | 0.000 | e2e_v5 基线（`bench/baseline_e2e.json`） |
| 2026-09-25 | **P3**（概念标签 + 情景兜底） | e2e_v5 test | **0.825** | 1.000 | 0.000 | **通过**：+0.032（按题聚类 CI [0.000, +0.087]，4 胜 0 负，边缘）；事实 0.635 → 0.698；合入 main |
| 2026-09-25 | main P3 | e2e_v6 test | 0.482 | 0.358 | 0.282 | e2e_v6 基线（`bench/baseline_e2e.json`；近似负例让拒答率很低） |
| 2026-09-25 | **P4**（无损更新 + 按概念找回） | e2e_v6 test | **0.563** | 0.358 | 0.308 | **通过**：+0.080（按题聚类 CI [+0.036, +0.128]，31 胜 4 负）；事实 0.560 → 0.720；合入 main |
| 2026-09-25 | main P4 | e2e_v7 test | 0.599 | 0.375 | 0.410 | e2e_v7 基线（`bench/baseline_e2e.json`） |
| 2026-09-25 | P5（按主语回忆） | e2e_v7 test | 0.729 | 0.844 | 0.462 | **未通过**：主指标 +0.130（CI [+0.073, +0.193]），但过时率 +0.051 > 0.05（39 个更新题次里多了 2 个过时） |
| 2026-09-25 | main P4 | e2e_v8 test | 0.451 | 0.472 | 0.542 | e2e_v8 基线（`bench/baseline_e2e.json`） |
| 2026-09-25 | **P5b**（按主语回忆 + 按概念取最新 + 扩充词表） | e2e_v8 test | **0.586** | **0.708** | **0.306** | **通过**：+0.135（按题聚类 CI [+0.076, +0.198]，61 胜 9 负）；更新题 0.375 → 0.639；合入 main |
| 2026-09-25 | main P5b | e2e_v9 test | 0.482 | 0.750 | 0.646 | e2e_v9 基线（`bench/baseline_e2e.json`） |
| 2026-09-25 | P6（规则承诺 + 一行轨迹） | e2e_v9 test | 0.531 | 0.625 | 0.646 | **未通过**：主指标 +0.049（CI [0.000, +0.102]），但拒答率 −0.125 > 0.05；轨迹 0.021 → 0.521 |

P1 在 e2e_v2 test 上分题型的结果（main → P1）：情景 0.067 → 1.000，承诺 0.000 → 0.667，时间 0.500 → 1.000，更新 0.333 → 0.778，事实 0.833 → 0.792，轨迹 0.000 → 0.167，负例 0.667 → 0.667。

实验过程和失败分析见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 计划中的评测

- `retrieval_v3`：更多玩家，加入"需要总结多条记忆"和"抽象问法"类查询（v2 暴露出的弱点）。
