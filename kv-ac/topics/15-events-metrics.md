# 15 · KV Events、指标与可观测性（W4-D26）

## 核心问题
1. 事件类型、发射点、外部消费方式？
2. hit rate / usage / 逐出指标从哪来？
3. 路由器（llm-d 等）怎么用这些信号？

## 代码索引
- `vllm/distributed/kv_events.py`（BlockStored 56/BlockRemoved 114/AllBlocksCleared 135/
  KVEventBatch 139/ZmqEventPublisher 304/Aggregator）
- `vllm/config/kv_events.py`；发射点 block_pool(345/375/448/547/613/866)、
  kv_cache_manager.take_events(716-741)、scheduler(2303-2318)
- 指标：kv_cache_metrics.py(46)、PrefixCacheStats(kv_cache_manager:236)、
  SchedulerStats(2852/2874)、KVConnectorProm(metrics.py)

## 种子结论
- BlockStored 携带 hash/parent_hash/token_ids/extra_keys/group_idx/kv_cache_spec_kind/
  sliding_window/medium——外部（路由器/缓存服务）可据此重建整个前缀树并做请求级 KV 感知路由。
- 命中重放事件（emit_cached_block_events）仅在 kv_cache_report_mode=full 时发（get_computed_blocks 触发）。
- 发布：scheduler.update_from_output 排水 take_events → KVEventBatch → ZMQ（endpoint/
  replay_endpoint/buffer_steps/hwm/max_queue_size 可配）。
- 指标链：PrefixCacheStats（每 schedule 记录、make_stats 重置返回）→ hit rate；
  BlockPool.get_usage → kv_cache_usage；KVCacheMetricsCollector ~1% 采样块寿命/空闲/
  复用间隔 + 逐出事件。
- 部署集成文档（llm-d/production-stack/dynamo）演示消费端姿势；ec_transfer 是 encoder
  cache 的平行框架；kv_hints 是编排器→引擎的 KV 提示通道（msgspec、版本化）。

## 沉淀区
