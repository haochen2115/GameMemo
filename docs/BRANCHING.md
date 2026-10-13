# 分支规范与 SOTA 门禁

## 原则

**正式分支只放当前最强（SOTA）的版本。** 任何改动想进 `main`，必须在固定评测集上证明自己比 `main` 上现有的系统更好，并且不能在任何守护指标上明显变差。其余工作都在语义清晰的分支上进行。

## 分支结构

本仓库按**研究创新点**分成两个正式分支，其余都是历史实验分支。

| 分支 | 内容 | 状态 |
|---|---|---|
| `main` | **GameMemory**：面向游戏陪伴的事件触发记忆（对局账本 + 原话日志 + 意图索引），以及它所依赖的原话记忆（P7 / P8）。论文 `docs/PAPER.md`，主页 https://haochen2115.github.io/GameMemo/game-memory/ | 正式版，默认分支 |
| `shared-memory` | **Shared Memory Is Harder**：多租户客服智能体的共享记忆（QUORUM 机制、COMMONS 基准、威胁模型）。主页 https://haochen2115.github.io/GameMemo/shared-memory/ | 正式版（研究预览：机制和基准规范已可执行，模型实验待做） |
| `gh-pages` | 项目主页的发布内容，由 `main` 上的 `site/` 生成 | 自动 |
| `archive/v0-keyword-baseline` | v0 的冻结快照 | 只读 |
| `exp/personal-memory-v2-foundation` | v2 检索与写入重构（原 `research/personal-memory-foundation`） | 历史，已合入 |
| `exp/personal-memory-p1-write-path` | 写入链路 + P1 | 历史，已合入 |
| `exp/personal-memory-p2-consolidation` | P2：按属性回忆、偏好门槛、接地检查 | 历史，已合入 |
| `exp/personal-memory-p3-facts` | P3：概念标签、情景兜底 | 历史，已合入 |
| `exp/personal-memory-p4-extraction` | P4：无损更新、按概念找回 | 历史，已合入 |
| `exp/personal-memory-p5-subjects` | P5 / P5b：按主语回忆 | 历史，P5b 已合入 |
| `exp/personal-memory-p6-promises` | P6 / P6b：承诺、轨迹 | 历史，P6b 已合入 |
| `exp/raw-memory-and-game-memory` | 回答层评测、原话记忆 P7–P10、LoCoMo + Mem0、GameMemory（E11–E21）的开发记录 | 历史，已合入 `main` |
| `exp/shared-memory-preview` | 共享记忆最初的研究预览（原 `research/shared-memory-quorum`） | 历史，已并入 `shared-memory` |

## 新分支命名

| 前缀 | 用途 | 生命周期 |
|---|---|---|
| `main` / `shared-memory` | 两个研究方向的正式版 | 永久 |
| `exp/<想法>` | 实验和开发分支 | 合入正式分支后保留作记录，或删除 |
| `archive/<内容>` | 冻结的历史快照 | 永久 |

分支名要说清楚"这是什么"，不用工具生成的随机名。

## SOTA 门禁（进入 main 的条件）

1. `pytest` 全部通过。
2. 在 `bench/candidate.json` 声明的每个评测上（检索看 `bench/sota.json`，端到端看 `bench/sota_e2e.json`，回答层看 `bench/sota_answer.json`；从 P7 起主门禁是回答层的 `e2e_v11` **test** split，E2EScore 和 `retrieval_v2` 只对"存改写后事实"的系统适用，见 BENCHMARK.md）：
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
