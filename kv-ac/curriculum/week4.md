# Week 4（D22-D30，9 天）分布式 KV + Maintainer 实战

> 周目标：PD 双实例真机跑通；提交 1 个 PR；mini connector 跑通；5 份 review 笔记；design map。
> 对应讲义：topics/14~15。所有 PR 活动遵守 AGENTS.md（重复性检查、uv、pre-commit、
> human submitter 署名、AI 声明）。

---

## D22 Connector 框架契约（逐钩子日）

**代码索引**
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py`（753 行，通读两遍）：
  KVConnectorBase_V1、KVConnectorRole(SCHEDULER/WORKER)、KVConnectorMetadata(S→W)、
  KVConnectorWorkerMetadata+aggregate(W→S)、KVConnectorHandshakeMetadata、
  KVConnectorTransferResults(finished_sending/finished_recving/failed_recving)、
  SupportsHMA+supports_hma、get_required_kvcache_layout(639)、requires_piecewise_for_cudagraph(658)、
  get_finished_count(679)、reset_cache(741)
- `kv_connector/factory.py`：KVConnectorFactory、create_connector（HMA 强制与
  `--disable-hybrid-kv-cache-manager` 提示）、注册表(153-249)、外部 connector 加载
  （kv_connector_module_path、第 3 参 kv_cache_config）
- `kv_transfer_state.py`：worker 侧单例 _KV_CONNECTOR_AGENT、engine_id 跨 TP/PP 广播
- `kv_connector/v1/metrics.py`：KVConnectorStats / Prometheus
- `vllm/config/kv_transfer.py` 全文（kv_role/kv_rank/kv_buffer_*、kv_load_failure_policy、
  enable_permute_local_kv、hisparse_host_pool_gib）

**Scheduler 侧调用点清单**（`vllm/v1/core/sched/scheduler.py`，挨个看上下文）
951（get_num_new_matched_tokens，块对齐本地命中作输入）｜1242（update_state_after_alloc，
async load 可能两次）｜1490-1523（build_connector_meta）｜2560（on_new_request）｜
2932-2985（_connector_finished→request_finished，返回 delay_free+kv_transfer_params，
2162-2168 消费）｜3123（update_connector_output）｜3244（_handle_failed_recving）｜
2308（take_events 合并）｜755/2750（has_pending_block_frees / has_pending_push_work）｜527-529（divergent hits）

**Worker 侧调用点清单**
`kv_connector_model_runner_mixin.py`：bind/clear metadata(77/106)、start_load_kv(85/90，
sync 由 has_sync_kv_loads 决定)、wait_for_save(92)、get_transfer_results(94)、
get_block_ids_with_load_errors(100)、stats/events/worker meta(102-104)、kv_connector_no_forward(25)
`gpu_model_runner.py`：register_kv_caches + set_host_xfer_buffer_ops(7394-7395)、
no_forward 入口(4214)、handle_preemptions(4180)
per-layer 流水线：`model_executor/layers/attention/kv_transfer_utils.py` 51/57
（maybe_transfer_kv_layer 装饰 wait_for_layer_load / save_kv_layer）

**驱动问题**
1. SCHEDULER/WORKER 两个 role 的实例分别由谁创建、生命周期挂在哪？
2. KVConnectorMetadata 与 KVConnectorWorkerMetadata 为什么方向不同、各自 aggregate 什么？
3. get_num_new_matched_tokens 三种返回值（(None,_) / (n,False) / (n,True)）分别怎么影响调度？
4. update_state_after_alloc 为什么可能被调两次？
5. delay_free=True 之后块的完整生命周期（谁持有、何时真正 free、fence 是什么）？

---

## D23 PD 分离端到端（多卡实验日）

**材料**：docs/features/disagg_prefill.md、docs/features/nixl_connector_usage.md、
docs/design/nixl_kv_push_connector.md、docs/design/nixl_kv_cache_lease.md、
examples/disaggregated/（disagg_proxy_demo.py、kv_events.sh、kv_load_failure_recovery_offline）

**代码索引**：`kv_connector/v1/nixl/`（connector.py 门面 + pull_scheduler/pull_worker、
push_scheduler/push_worker、base_scheduler/base_worker、tp_mapping、heartbeat、metadata）

**实验**（labs/README §2 模板）
1. 双卡/双实例：P(kv_producer) + D(kv_consumer)，NixlConnector UCX，跑通并观察 kv 传输日志
2. 切 push 模式对比（P 写 D 内存 vs D 读 P）；改 kv_load_failure_policy=review 语义
3. 观察 P 侧块释放时机（finished_sending 回报后）与 lease 续约日志

**驱动问题**
1. pull 与 push 的块释放安全性差异？heartbeat/lease 防什么（PR #41383 背景）？
2. requires_kv_delivery(192)：带 pending handoff 的请求被 preempt 时为什么选择重算而非交接？
3. handshake metadata 的 PP-aware keyed (pp_rank,tp_rank) 解决什么？
4. stateless_coordinator.py 的角色（跨实例进程组）？
5. do_remote_prefill / remote_prefill_cached_tokens 怎么避免 D 侧重算已传前缀？

**验收**：PD 实验记录写入 labs/，含一次失败注入（kill P）的观察。

---

## D24 Offloading 与分层缓存

**材料**：docs/features/kv_offloading_usage.md、docs/design/hisparse.md

**代码索引**
- `kv_connector/v1/offloading_connector.py` + `offloading/` 子包（scheduler/worker/common/
  **canonical_mapping**/events/metrics/config）
- 底座：`vllm/v1/kv_offload/`（config.py 归一化、OffloadingSpecFactory、tiering/ 的
  fs/obj/p2p/kvcr 分层、backpressure、async_lookup）
- 极简对照：SimpleCPUOffloadConnector + `vllm/v1/simple_kv_offload/`（manager/worker/
  copy_backend/cuda_mem_ops/disk_backend）
- 实验性稀疏：HiSparseConnector + `vllm/v1/hisparse/`（binding/layout/coordinator/block_pool/runtime）

**驱动问题**
1. offloading 的「块」从 GPU 池到 CPU/磁盘：地址映射谁维护？canonical_mapping 为什么是唯一
  TP/DCP/PCP-aware 的模块？"canonical pages" 抽象是什么？
2. 命中后 promotion（回 GPU）的路径？async_lookup 与调度怎么配合（backpressure）？
3. offloading/config.py:182 为什么把 V2 runner 布局排除在 parallelism-agnostic 之外？
4. simple_kv_offload 与 offloading 的取舍（教学/最小路径 vs 全功能）？
5. tier 的 spec（CPUOffloadingSpec 等）怎么配置扩展（spec_module_path）？

**实验**：`--kv-offloading-size` + OffloadingConnector 跑 e2e，制造重复前缀观察 CPU tier 命中与回迁

---

## D25 Connector 群像 + HMA 契约

**代码索引（按序精读→略读）**
1. **ExampleConnector**（教学模板，精读：自写 connector 就照它）
2. **MultiConnector**（load 取第一个广告命中的、save 全写、HMA 透传条件）
3. **NixlConnector**（D23 已读）→ 4. **OffloadingConnector**（D24 已读）
5. LMCacheConnectorV1（外部适配器 lmcache_integration/vllm_v1_adapter.py、PIECEWISE cudagraph）
6. MooncakeConnector（P2P RDMA）与 MooncakeStoreConnector（共享池 + 跨实例 hash 去重，
   自带 coordinator/worker/protocol）
7. HF3FSKVConnector（3FS 文件系统后端）、FlexKV、MoRIIO（ROCm）、DecodeBenchConnector
- HMA：docs/design/hybrid_kv_cache_manager.md + base.py SupportsHMA + test_hma_auto_config.py

**驱动问题**
1. ExampleConnector 的 metadata 结构与生命周期钩子实现顺序？
2. MultiConnector 的「load 优先级」由什么决定（config 顺序）？失败回退语义？
3. HMA 契约为什么出现？request_finished_all_groups 与旧 request_finished 的差异（多 group 释放语义）？
4. 哪些 connector 是 HMA-capable？不满足时用户的出路（--disable-hybrid-kv-cache-manager）？
5. Mooncake Store 与 OffloadingConnector 的去重层次差异（跨实例 vs 实例内）？

**产出**：connector 速查表写入 topics/14 沉淀区（名称/模式/后端/何时用）。

---

## D26 KV Events 与 Encoder-Cache 对照

**代码索引**
- `vllm/distributed/kv_events.py`：BlockStored(56)/BlockRemoved(114)/AllBlocksCleared(135)/
  KVEventBatch(139)/KVEventAggregator、ZmqEventPublisher(304)、EventPublisherFactory(564)
- `vllm/config/kv_events.py`（publisher/endpoint/replay_endpoint/buffer_steps/hwm/topic）
- 发射点复习：block_pool 345/375/448/547/613/866；kv_cache_manager take_events(716-741) 的
  per-group 注记；scheduler 2303-2318 排水与发布
- 对照框架：`vllm/distributed/ec_transfer/`（encoder cache 的 E/PD 分离，/dev/shm tier、
  NIXL P2P pull）、`vllm/distributed/aux_output_connector/`、`vllm/v1/kv_hints/protocol.py`
- 集成视角：docs/deployment/integrations/（llm-d、production-stack、dynamo——路由器怎么消费事件）

**驱动问题**
1. BlockStored.parent_hash 让外部消费者能重建什么结构？kv_cache_spec_kind/sliding_window 注记给谁用？
2. replay_endpoint 与 KVEventAggregator 的用途（新实例预热？路由器重放？）
3. 事件满队列（max_queue_size/hwm）时行为？
4. ec_transfer 为什么不直接复用 kv_transfer（encoder cache 的什么特性不同：非分页？生命周期短？）
5. kv_hints 协议（KvHintsEnvelope）想解决什么（编排器下发 KV 提示）？

---

## D27 失败与可靠性

**代码索引**
- scheduler：_handle_failed_recving(3244)、kv_load_failure_policy（recompute|fail）、
  _handle_invalid_blocks(3259/3316) → BlockPool.evict_blocks
- worker：get_block_ids_with_load_errors(mixin:100) → KVConnectorOutput.invalid_block_ids
- 测试精读：test_error_propagation.py、test_kv_load_failure_recovery.py、
  test_invalid_blocks_correctness.py、test_cache_pollution_prevention.py、
  test_remote_prefill_lifecycle.py / test_remote_decode_lifecycle.py、
  test_bidirectional_kv_transfer.py、test_output_aggregator.py、test_handshake_pp_aggregation.py

**驱动问题**
1. 哪些真实场景会产生 load 失败（网络/对端重启/lease 过期/块被复用）？
2. recompute 策略的块级语义：invalid 块怎么作废、为什么必须 evict 而不是只重算 token？
3. cache pollution 防御：test_cache_pollution_prevention 防的正是哪种写坏缓存的路径？
4. KVOutputAggregator 与 get_finished_count 怎么配合（PP/多 worker 聚合）？
5. 双向传输（bidirectional）测试覆盖的场景是什么？

---

## D28 实战一：提一个真 PR

1. 选题材：`gh issue list --repo vllm-project/vllm --label kv-cache`（或 good first issue）/
   或从你学习中发现的小 bug、测试缺口
2. 重复性检查（AGENTS.md 强制）：`gh pr list --state open --search "<关键词> in:body"`
3. 实现 + `pre-commit run` + 相关 pytest 全绿 + （若影响输出/精度）模型 eval 结果
4. PR 描述：动机/方案/测试命令与结果/「本 PR 使用 AI 辅助」声明/**由你本人提交并署名**
5. 提交后把链接与 CI 状态记入 reviews/README.md

（本仓库 fork 的提交规范：Co-authored-by trailer；确保你理解并认可每一行改动）

---

## D29 实战二：mini Connector

- 以 ExampleConnector 为模板写一个「本地文件系统 KV 落盘」connector：
  - 实现 get_num_new_matched_tokens / update_state_after_alloc / build_connector_meta /
    start_load_kv / wait_for_save / request_finished 六个核心钩子
  - metadata 设计：块 hash→文件路径映射
- 验证：注册进 factory（kv_connector_module_path 外挂即可），跑
  tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py 的思路写一个最小用例
- 产出：代码 + 用例放 labs/mini_connector/（不提交 vllm 仓库），笔记写入 reviews/

---

## D30（10-27 周二前后）收官日

1. **Review 冲刺**：`gh pr list --repo vllm-project/vllm --state open --search "kv_cache"`，
   挑 5 个（覆盖 core/kernel/connector 至少各 1），逐个写 review 笔记（先自评，再与我对照）
2. **Design Map**：30 分钟不看资料画一页纸模块地图 → 存 reviews/design-map.md → 与
   topics/00-code-map.md 对照查漏
3. `/kv-quiz final`（跨四周 15 题，目标 ≥13）
4. `/kv-progress` 终盘点 + 遗留问题清零 + 下季度跟进清单（PR 的 review 回复、issue 认领）

**周验收**：里程碑 4 达成 → Maintainer Bar 自评。
