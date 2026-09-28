# 09 · Prefix Caching 与 Hash 链（W2-D9/D10）

## 核心问题
1. hash 链设计（parent 链、extra_keys、group id 打包）？
2. 三套 size（scheduler/hash/group block size）怎么来的？
3. 查找流程与 partial hit？

## 代码索引
- `vllm/v1/core/kv_cache_utils.py`：BlockHash(62)/WithGroupId(75)/init_none_hash(161)/
  extra_keys(611)/hash_block_tokens(650)/resolve_kv_cache_block_sizes(705)
- `vllm/v1/request.py` update_block_hashes(272/285)；`vllm/v1/engine/core.py`(230-236)
- `vllm/v1/core/kv_cache_manager.py` get_computed_blocks(264)
- `vllm/v1/core/kv_cache_coordinator.py` 固定点(833)；FullAttentionManager(742)
- `vllm/v1/core/block_pool.py` cache_full_blocks(225)/cache_partial_block(448)

## 种子结论
- hash_block_tokens = H(parent_hash‖token_ids‖extra_keys)：链式使得相同前缀必然同 hash，
  一次比较即可确认整段前缀（防哈希碰撞语义上仍靠 token 复核的隐含假设 + extra_keys 收窄）。
- extra_keys 四类：LoRA 名、多模态（identifier+块内 offset）、cache_salt（只放第一块，
  租户隔离不必逐块携带）、prompt-embeds SHA。
- BlockHashWithGroupId 把 4 字节 group id 打进 hash bytes：同一逻辑前缀在不同 group
  （物理页大小/管理器不同）下天然不同键，杜绝跨组误命中。
- 算法选择：sha256/sha256_cbor 用确定性种子（可跨进程/跨节点复现，事件可重建）；
  xxhash 默认随机种子（快，但 NONE_HASH 语义不同）。
- 粒度：hash_block_size = prefix_match_unit 或各可缓存组 block_size 的 GCD；
  scheduler_block_size = 各组 LCM；查找上限 num_tokens-1（至少留 1 token 现算）。
- partial hit（fine-grained）：block_size>hash_block_size 时内部探测多个 hash 边界，
  尾部 partial 块登记靠 cache_partial_block + CoW 保护共享尾块。

## 沉淀区
