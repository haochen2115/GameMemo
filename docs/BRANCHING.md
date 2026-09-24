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
| `main` | 个人记忆 v2：混合检索（BM25 + jina-zh）、校验后的写入链路、双时态更新 | 当前 SOTA（retrieval_v2 test MemScore@3 0.892） |
| `archive/v0-keyword-baseline` | v0 的冻结快照（= 旧 main @ `3a866ba`） | 只读 |
| `research/personal-memory-foundation` | 产出当前 SOTA 的研究分支 | 已合入 main |
| `research/shared-memory-quorum` | 多租户共享记忆（QUORUM / COMMONS） | 研究预览，尚无模型评测 |

## SOTA 门禁（进入 main 的条件）

1. `pytest` 全部通过。
2. 在 `bench/sota.json` 指定的数据集和 split（目前是 `retrieval_v2` 的 **test** split）上：
   - 主指标 `MemScore@3` 至少比记录的 SOTA 高 `min_gain`（0.02）；
   - 守护指标 `Recall@3`、`MRR`、`Abstain` 的下降幅度都不超过各自的容忍度。
3. 用脚本检查，不靠人工判断：
   ```bash
   python -m bench.run_retrieval --split test --out bench/results/candidate.json
   python -m bench.gate bench/results/candidate.json --system <系统名>
   ```
   CI 会在所有指向 `main` 的 PR 上自动跑这一步（`.github/workflows/ci.yml` 的 `sota-gate`），对比的是 **base 分支**上的 `bench/sota.json`。
4. 合入的同一个 PR 里，更新 `bench/sota.json`（新 SOTA 的数字和 ref），并在 `docs/BENCHMARK.md` 的排行榜追加一行。
5. 候选系统要在跑 test 之前预登记（写进 commit message），一次只登记一个。

## 评测纪律

- **只在 dev split 上调参。** test split 用于最终报告；每看一次 test 都要在 BENCHMARK.md 里记录。
- 数据集升级（例如 `retrieval_v2`、端到端 QA 评测）时，旧 SOTA 要在新数据集上重跑，之后再切换 `sota.json`。
- test 集要在任何系统跑之前单独提交封存；最终报告附上配对 bootstrap 置信区间。
- 如果门禁挡住了一个整体更好的改动（比如主指标大涨、但一个守护指标小跌），不要放宽门禁，而是修掉回退，或者在 PR 里写明理由，交给维护者决定。
