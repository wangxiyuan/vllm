# 14 · KV Connector 框架与 PD 分离（W4-D22~D25/D29）

## 核心问题
1. 契约钩子全表与调用时机？
2. PD 一次请求的完整时序？
3. pull/push/offload 各 connector 的取舍？HMA 契约？

## 代码索引
- 契约：`kv_connector/v1/base.py`（753 行）；factory(153-249)；kv_transfer_state.py；metrics.py
- 调用点：见 00-code-map §7 与 curriculum/week4.md D22 清单
- connector：nixl/（pull·push+heartbeat+lease）、offloading_connector+kv_offload/、
  MultiConnector、SimpleCPUOffload、LMCache(×2)、Mooncake(×2)、HF3FS、FlexKV、MoRIIO、
  HiSparse、DecodeBench、ExampleConnector（自写模板）

## 种子结论
- 双 role：SCHEDULER 实例由 scheduler 持有（管理"块的可获得性"），WORKER 实例由
  runner mixin 持有（执行传输）。每步 S→W 走 KVConnectorMetadata（scheduler_output 携带，
  mixin bind），W→S 走 KVConnectorWorkerMetadata.aggregate 汇报完成度。
- 关键钩子语义：get_num_new_matched_tokens 三态 ((None,_) 推迟 / (n,False) 同步 / (n,True)
  异步)；update_state_after_alloc 决定拉取内容（async 场景可两次）；request_finished 返回
  (delay_free, kv_transfer_params)——delay_free 让块在传输完成前不被复用，靠
  finished_sending 回报 + has_pending_block_frees 维持调度循环。
- PD 时序（D 侧视角）：schedule 调 get_num_new_matched_tokens → 分配 → update_state_after_alloc
  → build_connector_meta → worker start_load_kv（sync 在 forward 前/async 并行，看
  has_sync_kv_loads）→ per-layer wait_for_layer_load → wait_for_save → get_transfer_results →
  scheduler update_connector_output → 命中失败走 kv_load_failure_policy(recompute|fail)。
- 可靠性：lease/heartbeat（NIXL）保证 P 侧块在 D 侧取完前不被释放；requires_kv_delivery
  让带未完成交接的请求 preempt 时走重算。
- HMA（SupportsHMA）：hybrid KV cache manager 下多 group 释放语义需
  request_finished_all_groups；factory 在创建时强制校验，不满足则要求
  --disable-hybrid-kv-cache-manager。
- canonical_mapping（offloading）：把 TP/DCP/PCP 分布下的页映射成 parallelism-agnostic
  的"canonical pages"，是唯一感知并行布局的模块；V2 runner 布局暂被排除（config.py:182）。

## 沉淀区
