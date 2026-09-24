# 分支规范与 SOTA 门禁

## 原则

**`main` 只放当前最强（SOTA）的版本。** 任何改动想进 `main`，必须在固定评测集上证明自己比 `main` 上现有的系统更好，并且不能在任何守护指标上明显变差。其余工作都在语义清晰的分支上进行。

## 分支命名

| 前缀 | 用途 | 生命周期 | 例子 |
|---|---|---|---|
| `main` | 当前 SOTA，默认分支 | 永久 | `main` |
| `research/<方向>-<里程碑>` | 一个研究方向上的一个里程碑，在这里开发、跑评测 | 合入 main 或放弃后归档 | `research/personal-memory-foundation` |
| `exp/<想法>` | 一次性小实验，不保证质量 | 用完即删 | `exp/bge-m3-embedder` |
| `archive/<内容>` | 冻结的历史快照，只读 | 永久 | `archive/v0-keyword-baseline` |

分支名要说清楚"这是什么"，不用工具生成的随机名（如 `claude/stoic-feynman-xxxx`、`codex/xxx`）。

## 当前分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `main` | 个人记忆 P1：混合检索、只读玩家发言、历史召回、情景记忆、助手承诺、回忆模式 | 当前 SOTA（retrieval_v2 0.892；e2e_v2 test 0.759） |
| `archive/v0-keyword-baseline` | v0 的冻结快照（旧 main @ `3a866ba`） | 只读 |
| `research/personal-memory-foundation` | v2 检索与写入重构 | 已合入（PR #1） |
| `research/personal-memory-write-path` | 写入链路 + P1 | 已合入（PR #2） |
| `research/personal-memory-consolidation` | 变化轨迹巩固、拒答 | 开发中 |
| `research/shared-memory-quorum` | 多租户共享记忆（QUORUM / COMMONS） | 研究预览，尚无模型评测 |

## SOTA 门禁（进入 main 的条件）

1. `pytest` 全部通过。
2. 在 `bench/candidate.json` 声明的每个评测上（检索看 `bench/sota.json`，端到端看 `bench/sota_e2e.json`；目前分别是 `retrieval_v2` 和 `e2e_v2` 的 **test** split）：
   - 至少一个评测标为 `improve`，并且主指标比记录的 SOTA 高 `min_gain`；其余评测标为 `hold`，主指标降幅不超过 `hold_tol`；
   - 所有守护指标（如 `Abstain`、`StaleRate`）变差的幅度都不超过各自的容忍度。
3. 用脚本检查，不靠人工判断：
   ```bash
   python -m bench.run_retrieval --split test --systems <候选检索系统> --out bench/results/candidate.json
   python -m bench.gate --candidate bench/candidate.json --base-dir <main 上的 SOTA 记录目录>
   ```
   CI 会在所有指向 `main` 的 PR 上自动跑这一步（`.github/workflows/ci.yml` 的 `sota-gate`），对比的是 **base 分支**上的 `bench/sota.json`。
4. 合入的同一个 PR 里，更新 `bench/sota.json`（新 SOTA 的数字和 ref），并在 `docs/BENCHMARK.md` 的排行榜追加一行。
5. 候选系统要在跑 test 之前预登记（`bench/candidate.json` 加上 commit message），一次只登记一个。候选没通过时，不能换一个 test 集重考同一个候选。
6. e2e 评测需要本地 LLM：基线和候选都跑 3 个 seed 并合并，结果文件提交进仓库，CI 会重新计分。
7. 评测换到新的封存数据集时，在分支里附上 `bench/baseline_*.json`（base 分支的代码在新数据集上的结果）；base 上的 SOTA 记录如果是另一个数据集的，门禁会自动改用这个基线。

## 评测纪律

- **只在 dev split 上调参。** test split 用于最终报告；每看一次 test 都要在 BENCHMARK.md 里记录。
- 数据集升级（例如 `retrieval_v2`、端到端 QA 评测）时，旧 SOTA 要在新数据集上重跑，之后再切换 `sota.json`。
- test 集要在任何系统跑之前单独提交封存；最终报告附上配对 bootstrap 置信区间。
- 如果门禁挡住了一个整体更好的改动（比如主指标大涨、但一个守护指标小跌），不要放宽门禁，而是修掉回退，或者在 PR 里写明理由，交给维护者决定。
