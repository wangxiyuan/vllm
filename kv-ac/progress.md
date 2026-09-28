# 学习状态机（唯一事实来源，导师每次会话先读这里）

## 当前位置
- 阶段：**W1 基础与全链路**
- 日：**D1**（2026-09-28 周一）
- 今日卡片：curriculum/week1.md → D1（理论基础 + KVCacheSpec 体系）

## 里程碑状态
| 里程碑 | 判定标准 | 状态 |
|---|---|---|
| W1 全链路图 | 手画"token 从 scheduler 到物理 KV 写入再读出"完整图 + num_blocks 手算验算误差为 0 + TP=2 跑通 | ☐ |
| W2 pool 状态机 | 徒手模拟 10 步 alloc/free/CoW 序列画出 pool 状态 + 新增 1 个 test_prefix_caching case 通过 + tests/v1/core 全绿 | ☐ |
| W3 backend 对比报告 | 3 个 backend 吞吐/精度对比 + 能逐行讲 reshape_and_cache_flash_kernel + kernel 单测 | ☐ |
| W4 实战产出 | 1 个 PR 提交 + mini connector 跑通 lifecycle 测试 + 5 份 review 笔记 + 一页纸 design map | ☐ |

## 验收明细（按日勾选，导师与学员核对后更新）
### W1
- [ ] D1 spec 层次默写 + KV 字节手算
- [ ] D2 num_blocks 验算误差 0
- [ ] D3 物理 block 布局图
- [ ] D4 写入路径讲到 CUDA 行
- [ ] D5 slot 手算无误
- [ ] D6 默认 backend 及理由
- [ ] D7 复盘测验 ≥12/15

### W2
- [ ] D8-D9 数据结构与 hash 链
- [ ] D10-D11 查找与分配全流程复述
- [ ] D12-D13 释放/淘汰/调度集成
- [ ] D14 tests/v1/core 全绿 + 新增 case + 测验 ≥12/15

### W3
- [ ] D15 两代 runner 对照结论
- [ ] D16 Mamba 混合管理讲透
- [ ] D17 量化 KV 两条流向图
- [ ] D18 CUDA graph 不变量复述
- [ ] D19 kernel 精读 + 单测
- [ ] D20 MLA 布局讲透
- [ ] D21 spec decode 回滚语义 + 测验 ≥12/15

### W4
- [ ] D22 connector 契约逐钩子
- [ ] D23 PD 双实例实验跑通
- [ ] D24 offloading 实验跑通
- [ ] D25-D27 群像/事件/可靠性
- [ ] D28 PR 提交
- [ ] D29 mini connector 跑通
- [ ] D30 5 份 review + design map + 终测 ≥13/15

## 薄弱点清单（测验错题与问答暴露，自动回流 /kv-quiz weak）
（暂无）

## 错题回流
（暂无）

## 下阶段重点
- 今日：完成 D1 卡片；环境准备见 labs/README.md
