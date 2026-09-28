# 11 · Hybrid 协调器与各 Manager（W2-D10/D12 + W3-D16）

## 核心问题
1. 固定点查找为什么需要、怎么收敛？
2. 各 SingleType manager 的差异要点？
3. Mamba/RSWA/SWA 的特殊行为？

## 代码索引
- `vllm/v1/core/kv_cache_coordinator.py`：三协调器(471/522/607)、verify_and_split(728)、
  固定点(833)、per-group(970)、cache_blocks 覆写(799，EAGLE +1 块)
- `vllm/v1/core/single_type_kv_cache_manager.py`：Full(739)/RSWA(900)/SWA(946)/
  Circular(1194)/KpoolTail(1283)/ChunkedLocal(1287)/Mamba(1443，checkpoint 1675-2031)/
  Cross(2122)/Sink(2187)/HiSparse(2211-2591)/工厂(2592)

## 种子结论
- Hybrid 查找 = 各组 hit 长度互相约束的不动点迭代：SWA 组的最大可服务长度依赖 full 组
  命中长度，反之亦然；"full attention downward-closed"（命中到第 k 块 ⇒ 前 k-1 块全命中）
  用于单调裁剪迭代空间。分歧时产出 shared_prefix_boundary（各组一致的最长公共前缀）。
- manager 差异速记：
  - Full：链式扫描 + fine-grained 内部探测 + partial tail 缓存
  - SWA：窗口滚动，块生命周期由 get_num_skipped_tokens 决定，null 块占位
  - RSWA：窗口内 gap（中间被逐出段）回收，支持重放边界
  - ChunkedLocal：chunk 边界对齐，窗口跨 chunk 合并
  - Mamba：状态不可部分共享 → checkpoint/align 模式 + partial hash
  - Cross：encoder 长度决定 slot，非因果
  - Sink：attention sink 常驻块
- EAGLE 组：cache_blocks 多缓存一块（lookahead），find_longest_cache_hit 末块 drop 验证。

## 沉淀区
