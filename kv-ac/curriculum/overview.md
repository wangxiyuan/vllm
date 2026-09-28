# 30 天总纲：KV Cache Maintainer 冲刺

## 终点线（Maintainer Bar）

1. 对任意 kv cache 相关 PR：**15 分钟内**判断它动了哪一层（core / worker / kernel / connector）、
   正确性风险点在哪、该跑哪些测试。
2. 能独立设计新 `KVCacheSpec` / 新 connector，并说清要触碰的每个契约点。
3. 社区流程肌肉记忆：重复性检查 → 实现 → pre-commit → 测试证据 → PR 描述（AI 声明、human submitter）。
4. 能把模块讲给别人听：一页纸 design map + 一份疑难 FAQ。

## 你手上的仓库（重要：与网上旧资料差异大）

2026-09 快照的 vLLM 已深度演进：
- `KVCacheManager` 拆为 **manager / coordinator / single_type_manager / block_pool** 四层；
  `cache_finished_request`/`cache_aborted_request` 已删除（现为 `finish_requests` + `_free_request`）。
- 注意力体系在 `vllm/v1/attention/`；老的 `v1/attention_backends/` 与 **v0 CUDA PagedAttention kernel 已移除**。
- 两代 model runner：V1 `vllm/v1/worker/gpu_model_runner.py`（~7.5k 行）vs MRV2 `vllm/v1/worker/gpu/model_runner.py`（~2.3k 行），`use_v2_model_runner` 切换。
- 新增 `KVCacheLayout` 五维布局系统（RFC #42082）、HMA connector 契约、15+ connector、
  KV 量化扩到 fp8/nvfp4/int8/int4/per-token-head。
- 规模：core 层 ~1.2 万行、attention backend ~1.5 万行、相关测试 ~2 万行——所以必须按周分主题推进。

## 四周地图

| 周 | 主题 | 核心代码 | 里程碑 |
|---|---|---|---|
| W1 (D1-7) | 基础与全链路：spec 体系、显存规划、物理布局、写入/读取路径、backend 选择 | kv_cache_interface / kv_cache_utils(容量) / worker utils / cache_kernels.cu / block_table / selector | 全链路图 + num_blocks 验算 |
| W2 (D8-14) | 块管理核心：BlockPool、hash 链、查找、分配、CoW、淘汰、preemption、调度集成 | block_pool / kv_cache_utils(哈希) / kv_cache_manager / coordinator / single_type / scheduler | pool 状态机 + 新增测试 |
| W3 (D15-21) | Worker 与 Kernel：双代 runner、Mamba、量化 KV、CUDA graph、kernel 精读、MLA、spec decode | gpu_model_runner(V1/V2) / mamba_utils / KVQuantMode / cudagraph / cache_kernels.cu / mla/ / eagle | backend 对比报告 |
| W4 (D22-30) | 分布式与实战：connector 契约、PD 分离、offloading、events、可靠性 + PR 实战 | kv_transfer/ / kv_offload/ / kv_events.py + gh 实战 | PR + mini connector + 5 review |

## 每日节奏（全职 8h）

```
09:00-12:00  主题块 A（读论文/文档 + 带着驱动问题读代码）
14:00-17:00  主题块 B（同上）
17:00-18:00  当日实验（labs/README.md）
18:00-18:30  自答校对驱动问题 + /kv-qa 清障 + journal 打卡
周日 = 复盘日：上午画图/整理笔记，下午 /kv-quiz 15 题判卷，/kv-progress 盘点
```

## 问题驱动学习循环（每天执行）

1. **先自答**：当日驱动问题不看代码先写答案（写进 topics/ 对应文件草稿区）。
2. **读码验证**：按卡片里的代码索引逐个验证，修正答案。
3. **答疑沉淀**：卡住的用 `/kv-qa`，结论自动进 qa/ 与 topics/。
4. **周测验回流**：错题进薄弱点清单，下周测验优先重考。

## 进度风险规则

- 落后 ≤1 天：压缩当日实验为「读日志+跑单测」级别，周末补。
- 落后 ≥2 天：砍 W3 的 MLA 深读（保留结论级）与 W4 的 connector 群像（保留 3 个代表），保 PR 实战不砍。
- 每次测验 <10/15：下一天上午改为错题主题回炉。
