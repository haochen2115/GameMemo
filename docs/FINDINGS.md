# 别让模型改写记忆：小模型长期记忆的写入损失

> **Don't let the model rewrite your memories.** For LLM agents with long-term memory on small models, storing the conversation verbatim and retrieving it beats extracting "facts" with the model — on our own Chinese companion benchmark and on the public LoCoMo benchmark against Mem0. The extraction step is where most of the information is lost, and evidence-level metrics hide it.

本文整理 GameMemo 2026-10-08 ~ 10-10 的实验（`docs/EXPERIMENTS.md` E11–E19）。所有数字都来自预登记的实验；本项目自己的结论在封存 test 集上得出，外部验证在公开的 LoCoMo 上得出。

## 结论

1. **改写式记忆的主要损失发生在写入时，而不是检索时。** 在 LoCoMo 上，Mem0（`qwen2.5:3b`）把 272 个 session、约 5900 句对话写成 1265 条记忆，21% 的 session 一条也没留下。标准答案能在原话里找到的题目中，**67% 的答案不在 Mem0 的记忆库里**（不依赖任何模型的字符串检查）；原话检索能答对的题目中，Mem0 有 **72%** 即使把全部记忆都给回答模型也答不出（7B 判官）。时间信息丢得最彻底：这类题 49 / 56 的日期没被记下来。
2. **原话检索在两个基准上都显著好于改写式记忆。** LoCoMo：`raw:10` 0.321 vs Mem0 0.140（+0.181，95% CI [+0.155, +0.207]）。本项目数据：原话记忆（P7 / P8）在三个封存 test 集上比改写式的 P6b 高 0.15–0.20；换 7B 回答模型后仍高 0.195。
3. **"把全部历史放进 prompt"只在短历史上可行，交叉点大约在 4k–10k token。** LoCoMo 前 5 个 session（约 4k token）时两者持平，12 个 session（约 1 万 token）时原话检索高 0.058，全长（约 2 万 token）时高 0.073。本项目数据上 3B 模型在 148 段对话时从 0.797 掉到 0.602，7B 从 0.852 掉到 0.695；检索几乎不受长度影响。
4. **只看"检索证据里有没有答案"的评测会把人带向错误的方向。** 本项目前 6 个版本（P1–P6b）都在优化改写式记忆，用的指标都显示在进步；换成判断回答本身、并加上"全部历史""原话检索"两个朴素基线后，才发现这条路线比什么都不改写还差。

## 1. 为什么改写会丢信息

改写式记忆（Mem0、MemGPT 类，以及本项目的 P1–P6b）在写入时让 LLM 读对话、抽出"值得记住的事实"，再决定 ADD / UPDATE / DELETE。这一步有三种损失：

- **漏写**：小模型只挑它认为重要的事。LoCoMo 上平均每 22 句对话只留下 4.7 条记忆，21% 的 session 什么都没留下，2.6% 的 session 因模型输出的 JSON 无法解析而整段丢失。
- **改写掉细节**：日期、数字、原话里的专有名词在抽成"事实"时被概括掉。"Caroline attended a charity race"——什么时候？LoCoMo 时间题上，Mem0 的记忆库几乎从不保留日期；本项目里 "我在服装厂负责跟单" 被写成 "在服装厂上班"。
- **记错**：Mem0 把 "yesterday" 换算成了**写入那天**的日期而不是对话发生的日期（我们在每个 session 开头写明日期后才避免）；P6b 时期为此专门加过"接地检查"。

原话记忆没有这一步：每句话带着时间原样保存，写入**不调用 LLM**（本项目 P6b 写一个 test 集要 867 次调用、194 分钟，原话记忆不到 1 分钟），损失只剩检索召回：LoCoMo 上证据句全部取回时答对 0.484，一句都没取回时 0.076。

| | 改写式（Mem0 / P6b） | 原话（`raw` / P7–P8） |
|---|---|---|
| 写入 | 每段对话 1–2 次 LLM 调用 | 0 次 |
| 写入损失 | LoCoMo 上可在原话找到的答案 67% 不在库里 | 无 |
| 主要失分 | LoCoMo：写入 72%；本项目：检索 56% | 检索召回 |
| 时间信息 | 大多丢失 | 每句自带日期，回答时换算 |
| 更新 / 冲突 | 写入时由 LLM 决定覆盖，可能误删 | 不覆盖，回答模型按时间判断 |

两种改写式记忆失败的位置不同（P6b 的记忆比较密，问题在检索；Mem0 的记忆很稀，问题在写入），但都输给了不改写的做法。

## 2. 外部验证：LoCoMo + Mem0（E19）

LoCoMo 10 段长对话（19–32 个 session，约 1.2–2.5 万 token），类别 1–4 共 1540 题；回答模型 `qwen2.5:3b`；判官 `qwen2.5:7b`（与 token F1≥0.5 的一致率 86–88%）。参数在运行前固定，没有在 LoCoMo 上调。

| 上下文 | J | 多跳 | 时间 | 开放 | 单跳 | prompt 字数 |
|---|---|---|---|---|---|---|
| **原话检索 `raw:10`** | **0.321** | **0.259** | **0.174** | 0.042 | **0.430** | 1.9k |
| 全部对话 `full` | 0.249 | 0.220 | 0.047 | 0.062 | 0.357 | 89k |
| Mem0 检索 10 条 | 0.140 | 0.206 | 0.056 | 0.052 | 0.161 | 1.2k |
| Mem0 全部记忆（不检索） | 0.092 | 0.156 | 0.044 | 0.031 | 0.095 | 14.6k |

配对 bootstrap：原话 − Mem0 **+0.181 [+0.155, +0.207]**（373 好 / 94 差）；原话 − 全部对话 +0.073 [+0.046, +0.099]；全部对话 − Mem0 全部记忆 +0.157 [+0.135, +0.180]。把 Mem0 的全部记忆塞进去反而比检索 10 条更差（−0.049）：对小模型来说，更多改写过的事实只是更多噪声。

长度曲线（只保留前 N 个 session 和证据都在其中的题）：

| 前 N 个 session | 题数 | 原话检索 | 全部对话 | 原话 − 全部 |
|---|---|---|---|---|
| 5（约 4k token） | 262 | 0.386 | 0.416 | −0.031 [−0.095, +0.034] |
| 12（约 1 万 token） | 585 | 0.366 | 0.308 | +0.058 [+0.015, +0.101] |
| 全部（约 2 万 token） | 1540 | 0.321 | 0.249 | +0.073 [+0.046, +0.099] |

## 3. 本项目数据上的同一规律（E11–E15）

中文游戏陪伴对话，回答层判分（模型真的回答，再按关键词 / 拒答规则判对错），每个候选都在由不看代码的 agent 编写、跑之前就封存的新 test 集上考：

| | e2e_v10（8 段） | e2e_v10（148 段） | e2e_v11 | e2e_v12 |
|---|---|---|---|---|
| 改写式 P6b | 0.594 | 0.560 | 0.633 | 0.547 |
| 全部历史 | **0.797** | 0.602 | 0.734 | 0.664 |
| 原话检索（无任何改进） | 0.711 | 0.695 | 0.695 | 0.633 |
| 原话记忆 P7 / P8 | 0.789（P8） | — | **0.781**（P7） | **0.727**（P8） |

- 把 P6b 的全部记忆放进 prompt（比原始对话还长）仍比原始对话低 0.16（E11b）：改写本身丢信息。
- 7B 回答模型下结论不变：P8 − P6b +0.195；7B 的"全部历史"同样在 148 段时崩（0.852 → 0.695）。

## 4. 原话记忆还差在哪里

- **检索召回是上限**：证据没取回就几乎答不对。LoCoMo 上 `raw:10` 只有 54% 的题取回了全部证据句。
- **"变化过程"类问题**：回答模型拿到十几句原话会概括成"从 A 变成了 B"。对段位、手机这类封闭属性，用模式从原话里按时间截出一行"黄金三（日期）→ 铂金二（日期）→ ……"能显著改善（P8，轨迹题 0.25 → 0.47）；对工作、住处这类开放属性，词表和读取时让 LLM 列状态都没用（E18），瓶颈在同一件事的不同阶段用词不同、检索不到一起。
- **时间推理**：3B 模型常把原话里的 "yesterday" 原样输出而不换算成日期；LoCoMo 时间题所有方法都低。
- **小改进很难测准**：每个封存集 128 题时 95% CI 约 ±0.04；P9、P10 在 dev 上 +0.03～+0.04，在封存集上都没过门槛（E16、E17）。

## 5. 给做记忆系统的人的建议

1. **先存原话，带时间。** 不要在写入时让模型改写、合并、覆盖。需要结构化时，做**指向原话的索引**（属性 → 提到它的原话），而不是用模型的转述替代原话。
2. **历史短于约 4k token 时直接全放进 prompt**；更长时用检索，交叉点因模型而异（7B 比 3B 晚）。
3. **评测要判回答，并且永远带上两个朴素基线**："全部历史"和"原话检索"。只在记忆系统的版本之间比较、只看检索证据，看不出整条路线是负收益。
4. **把"写入损失"和"检索损失"分开量**：把系统的全部记忆给回答模型（不检索）是一个便宜的探针；再用字符串检查答案是否还在记忆库里。

## 局限

- 回答模型是 3B / 7B，在 CPU 上运行；结论是否适用于大模型（GPT-4 级别的抽取质量高得多）没有验证。Mem0 论文报告的结果用的是 GPT-4o-mini。
- LoCoMo 上 Mem0 每个 session 调用一次 `add()`（官方评测是每两句一次，成本约 10 倍）；更细的写入粒度可能减少漏写。
- 判官是 7B 模型，单次运行，没有多 seed。
- 本项目的数据由 agent 编写，是虚构的中文游戏玩家；LoCoMo 也是合成对话。
- LoCoMo 类别 5（对抗题）按惯例排除。

## 复现

```bash
pip install -e '.[dev]' mem0ai ollama
ollama pull qwen2.5:3b && ollama pull qwen2.5:7b && ollama pull nomic-embed-text
# 长 prompt 时限制 llama-server 的主机端缓存，否则会因内存不足被杀
LLAMA_ARG_CACHE_RAM=1024 ollama serve
# LoCoMo（数据：snap-research/locomo 的 locomo10.json）
python -m bench.locomo answer --data locomo10.json --contexts raw:10,full --out answers.json
python -m bench.locomo mem0-ingest --data locomo10.json --out-dir bench/.mem0
python -m bench.locomo answer --data locomo10.json --contexts mem0:10,mem0all --mem0-dir bench/.mem0 --out answers.json
python -m bench.locomo judge --answers answers.json --judge qwen2.5:7b
python -m bench.locomo_analysis --all answers.json
# 本项目数据的回答层评测
python -m bench.run_answer --data bench/data/e2e_v11.json --split test --contexts none,full,rag:8 --seeds 0
```

结果文件：`bench/results/locomo/`、`bench/results/answer_*.json`、`bench/results/e12/`。
