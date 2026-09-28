# KV Cache 学习知识库（vLLM Maintainer 冲刺 · 30 天）

> 目标：一个月内达到 KV cache 模块 maintainer 水平——能独立 review/修复/设计，熟悉社区流程。
> 起点：2026-09-28（周一）。节奏：全职 8h/天。硬件：多卡 2-8 张。
> 注意：知识库中所有 `file:line` 行号基于仓库快照 **commit `027b6f3a22`**（2026-09-28）。
> 自查是否过期：`git rev-parse --short HEAD` 若不再是 `027b6f3a22`，说明代码已前进，
> 让导师（我）重新核对受影响的行号与结论（提交数少时通常只影响个别行号）。

## 目录导航

| 路径 | 内容 | 谁来更新 |
|---|---|---|
| [curriculum/overview.md](curriculum/overview.md) | 30 天总纲、达标线、方法论 | 导师（我） |
| [curriculum/week1.md](curriculum/week1.md) ~ [week4.md](curriculum/week4.md) | 逐日学习卡片（目标/材料/代码/驱动问题/实验/验收） | 导师 |
| [topics/00-code-map.md](topics/00-code-map.md) | 全模块代码地图（勘察成果） | 导师 |
| [topics/](topics) | 15 个主题讲义（随学习与问答充实） | 双方 |
| [qa/inbox.md](qa/inbox.md) | 问答原始沉淀 | `/kv-qa` 自动写入 |
| [qa/archive/](qa/archive) | 每周归档提炼的 FAQ | `/kv-progress` 归档 |
| [quizzes/bank.md](quizzes/bank.md) | 84 道分级自测题库（含参考要点） | 导师 |
| [quizzes/](quizzes) | 每次测验卷与错题本 | `/kv-quiz` 自动写入 |
| [labs/README.md](labs/README.md) | 实验环境手册 + 实验记录 | 双方 |
| [reviews/](reviews) | PR review 笔记、design map | `/kv-review` 与你 |
| [journal.md](journal.md) | 每日一句话学习日志 | 你（我可代填） |
| [progress.md](progress.md) | 学习状态机（进度/薄弱点/下阶段） | 我每次会话维护 |

## 怎么用（每天）

1. 打开当周 `curriculum/weekN.md` 的当日卡片，先自答「驱动问题」，再带着问题读代码。
2. 卡住了随时提问：**建议用 `/kv-qa <问题>`**（自动沉淀进 qa/ 并更新薄弱点）；直接聊天问也行，说一句「记下来」即可归档。
3. 做当日实验（环境见 labs/README.md），结果记到 labs/。
4. 睡前在 journal.md 写一行打卡。
5. 每周日复盘日：`/kv-quiz weekN` 出 15 题判卷 → `/kv-progress` 盘点 → 我调整下周计划。

## 自定义命令

| 命令 | 作用 |
|---|---|
| `/kv-qa <问题>` | 结合源码回答（带 file:line）→ 沉淀 qa/ → 更新薄弱点 |
| `/kv-quiz [week1\|week2\|week3\|week4\|weak\|final]` | 出 15 题 → 判卷 → 错题本 → 更新进度 |
| `/kv-progress` | 进度盘点、更新仪表盘、下阶段建议 |
| `/kv-review <PR号\|diff>` | 我当模拟 reviewer，输出 review 笔记 |

## 进度仪表盘

| 指标 | 状态 |
|---|---|
| 当前位置 | **W1-D1**（2026-09-28） |
| 已完成天数 | 0 / 30 |
| 里程碑 | W1 ☐ 全链路图 · W2 ☐ pool 状态机 · W3 ☐ backend 对比报告 · W4 ☐ PR+mini connector+5 review |
| 薄弱点 TOP | （暂无，随测验生成） |

> 仪表盘由 `/kv-progress` 维护；progress.md 是唯一事实来源。
