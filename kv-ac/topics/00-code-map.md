# 00 · KV Cache 全模块代码地图（快照 commit `027b6f3a22`，2026-09-28）

> 所有路径相对仓库根 `/home/wxy/code/vllm`。行号已对照该 commit 逐一验证；
> 若 `git rev-parse --short HEAD` ≠ `027b6f3a22`，行号可能漂移，让导师重核。
> 用法：W1 读 §2/§3/§5，W2 读 §1/§4，W3 读 §3/§5/§6，W4 读 §7。

## 0. 一图流：一个 token 的 KV 生命周期

```
请求 → Scheduler.schedule()  [v1/core/sched/scheduler.py:557]
 ├─ KVCacheManager.get_computed_blocks()         查前缀命中 [kv_cache_manager.py:264]
 │   └─ KVCacheCoordinator(NoPrefix|Unitary|Hybrid)          [kv_cache_coordinator.py]
 │       └─ SingleTypeKVCacheManager.find_longest_cache_hit()
 │           └─ BlockPool.get_cached_block()  hash 查表     [block_pool.py:197]
 ├─ (connector) get_num_new_matched_tokens                    [scheduler.py:951]
 ├─ KVCacheManager.allocate_slots()               分配闸门   [kv_cache_manager.py:371]
 │   ├─ remove_skipped_blocks()  SWA/RSWA 窗口回收
 │   ├─ get_num_blocks_to_allocate() → 容量检查(水位) → 失败则 preempt 循环
 │   ├─ allocate_new_computed_blocks()  two-phase touch    [coordinator.py:229]
 │   ├─ allocate_new_blocks()  新块 + CoW 重定向            [single_type:364/449]
 │   └─ cache_blocks() → BlockPool.cache_full_blocks()  挂 hash+事件 [block_pool.py:225]
 ├─ (connector) build_connector_meta / update_state_after_alloc
 └─ SchedulerOutput{block_ids,...} → Worker
Worker  [V1 gpu_model_runner.py | V2 gpu/model_runner.py，use_v2_model_runner 切换]
 ├─ BlockTables.compute_slot_mappings()  slot=block*bs+off  [gpu/block_table.py:195]
 ├─ AttentionMetadataBuilder.build() → CommonAttentionMetadata [attention/backend.py:596/386]
 └─ execute_model → 每层 Attention.forward                  [layers/attention/attention.py:488]
     ├─ torch.ops.vllm.unified_kv_cache_update  写 KV
     │   └─ impl.do_kv_cache_update → ops.reshape_and_cache_flash
     │       └─ cache_kernels.cu:326  slot→(block,offset) 写入（可量化）
     └─ torch.ops.vllm.unified_attention_with_output  读 KV
         └─ backend kernel 消费 block_table_tensor（页表）
请求结束 → finish_requests → _free_request → kv_cache_manager.free
     [scheduler.py:2628]（deferred free 可能延迟）→ BlockPool.free_blocks（LRU/LIFO 顺序）
```

## 1. Scheduler 核心（W2 主战场，~1.2 万行）

| 文件 | 行数 | 职责 |
|---|---|---|
| `vllm/v1/core/kv_cache_manager.py` | 956 | 调度器唯一 KV 入口 KVCacheManager；KVCacheBlocks 包装 |
| `vllm/v1/core/kv_cache_coordinator.py` | 1063 | 组协调：NoPrefix/Unitary/Hybrid；hybrid 固定点查找(833)；EAGLE 组 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 2719 | 每种注意力一个 manager（Full/RSWA/SWA/ChunkedLocal/Mamba/Cross/Sink/Circular/KpoolTail/HiSparse）+ 注册工厂(2592) |
| `vllm/v1/core/block_pool.py` | 903 | 物理块池：free 队列、hash 映射、事件、CoW 辅助、null block |
| `vllm/v1/core/kv_cache_utils.py` | 2887 | KVCacheBlock/FreeQueue/哈希链/分组/容量规划 get_kv_cache_configs(2628) |
| `vllm/v1/core/kv_cache_metrics.py` | 96 | ~1% 采样的块驻留/逐出指标 |
| `vllm/v1/core/encoder_cache_manager.py` | 428 | 多模态 encoder 输出缓存（独立于 KV 块池） |
| `vllm/v1/core/sched/scheduler.py` | 3339 | 主调度循环；所有 connector 调用点所在 |
| `vllm/v1/core/sched/async_scheduler.py` | 78 | 异步调度变体（output placeholder、每步缓存） |
| `vllm/v1/core/sched/output.py` | 337 | SchedulerOutput / NewRequestData（block_ids 下发） |

## 2. Spec / 接口层（W1-D1/D3）

| 文件 | 行数 | 职责 |
|---|---|---|
| `vllm/v1/kv_cache_interface.py` | 1585 | KVCacheSpec 全家（156/485/549/656/739/783/823/882/922/1032/1149/1156/1166/1223）+ KVCacheGroupSpec(1436) + KVCacheConfig(1457) + 布局计算(295/314/353) |
| `vllm/v1/kv_cache_layout.py` | 58 | KVCacheLayout 五维 [L,B,H,N,C] 排列族（RFC #42082） |
| `vllm/v1/kv_cache_spec_registry.py` | 219 | spec→manager 注册与校验 |
| `vllm/distributed/kv_events.py` | 596 | BlockStored/Removed/Cleared + ZMQ 发布 |
| `vllm/v1/request.py` | 412 | Request.block_hashes 增量链(225/272/285) |

## 3. Worker 侧（W1-D2/D3/D5，W3-D15/16）

| 文件 | 行数 | 职责 |
|---|---|---|
| `vllm/v1/worker/gpu_worker.py` | 1601 | determine_available_memory(571)；V1/V2 选择(510-526) |
| `vllm/v1/worker/gpu_model_runner.py` | 7528 | V1 巨石：slot(4065)/execute(4149)/初始化链(6966-7421) |
| `vllm/v1/worker/gpu/model_runner.py` | 2345 | V2/MRV2 runner（gpu/ 包：block_table、input_batch、spec_decode、cudagraph_utils） |
| `vllm/v1/worker/utils.py` | 795 | allocate_kv_cache(389) flat int8+as_strided；prepare_kernel_block_sizes(458)；bind_kv_cache(591) |
| `vllm/v1/worker/block_table.py` | 548 | V1 块表/slot（BlockTable 57、kernel 398） |
| `vllm/v1/worker/gpu/block_table.py` | 366 | V2 块表/slot（Triton kernel 242/282） |
| `vllm/v1/worker/gpu_input_batch.py` / `gpu/input_batch.py` | 1146/713 | 持久批（两代） |
| `vllm/v1/worker/mamba_utils.py` | 1687 | 混合模型状态：分组(689)、spec-decode context(770)、对齐拷贝 |
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | — | connector worker 侧全部钩子入口 |

## 4. Attention Backend 体系（W1-D6，W3-D18/20）

| 文件 | 职责 |
|---|---|
| `vllm/v1/attention/backend.py` (1152) | Backend ABC(58)/CommonAttentionMetadata(386)/Builder(596)/CGSupport(559)/Impl(907)/MLA 基类(1037) |
| `vllm/v1/attention/selector.py` (247) | get_attn_backend(105)、backend_per_kind |
| `vllm/v1/attention/backends/registry.py` (327) | AttentionBackendEnum(34) |
| `vllm/v1/attention/backends/flash_attn.py` (2126) | FA 后端 + cascade(1630) + DCP |
| `vllm/v1/attention/backends/flashinfer.py` (2803) | SM100 默认；HND 强制(2503-2506) |
| `vllm/v1/attention/backends/triton_attn.py` (849) | layout-agnostic |
| `vllm/v1/attention/backends/flex_attention.py` (1492) | 仅 LBNHC(130-133) |
| `vllm/v1/attention/backends/mla/` (~25 文件) | FlashMLA/CUTLASS/FlashInfer(+sparse)/Triton MLA、indexer |
| `vllm/v1/attention/backends/mamba*_attn.py 等` | 线性注意力/SSM 元数据 builder |
| `vllm/v1/attention/backends/composite.py` | 多 group 组合 backend |
| `vllm/v1/attention/backends/utils.py` (1193) | 布局解析(259)、虚拟批次(502)、decode/prefill 拆分(724)、cascade |
| `vllm/model_executor/layers/attention/attention.py` (801) | Attention 层与 custom-op 分发(488/724/767) |

## 5. Kernel（W1-D4/D5，W3-D19/20）

| 文件 | 职责 |
|---|---|
| `csrc/libtorch_stable/cache_kernels.cu` | reshape_and_cache_flash_kernel(326)、concat_and_cache_mla(414/462/516)、launcher(820/917) |
| `csrc/libtorch_stable/cache_kernels_fused.cu` / `nvfp4_kv_cache_kernels.cu` | 融合写入 / nvfp4 |
| `vllm/v1/attention/ops/triton_unified_attention.py` | Triton 主力读写 kernel |
| `vllm/v1/attention/ops/` 其余 | decode/prefill/cascade merge/cp·dcp·pcp 公共 |
| `vllm/v1/attention/ops/paged_attn.py` | v0 遗留 wrapper（CUDA 版已删；CPU/XPU/ROCm 用） |

## 6. 量化 / CUDA Graph / Spec Decode（W3）

- 量化：KVQuantMode(kv_cache_interface:39)、cache_dtype 值域(config/cache.py:39-57)、
  scale buffer(attention.py:136-153)、写侧 kernel 分支(cache_kernels.cu)、读侧 k/v_descale
- CUDA graph：AttentionCGSupport(backend.py:559)、cudagraph_utils.py(142)、
  block_table 地址稳定注释(182-238)、PAD_SLOT_ID
- Spec decode：v1/spec_decode/eagle.py、v1/worker/gpu/spec_decode/（eagle/mtp/rejection_sampler）、
  build_for_drafting(backend.py:735)、拒绝回滚(gpu_model_runner:5127-5225 + scheduler:2057-2066)

## 7. Connector / Offloading / Events（W4）

| 位置 | 内容 |
|---|---|
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` (753) | 契约与全部钩子 |
| `kv_connector/factory.py` | 注册表(153-249) + HMA 强制 |
| `kv_transfer_state.py` | worker 单例、engine_id |
| `kv_connector/v1/nixl/` | Nixl pull/push（scheduler/worker/heartbeat/lease） |
| `kv_connector/v1/offloading_connector.py` + `vllm/v1/kv_offload/` | 分层 offload（canonical_mapping 唯一 TP/DCP/PCP-aware） |
| `kv_connector/v1/` 其余 | LMCache/LMCacheMP/MultiConnector/SimpleCPUOffload/HiSparse/MoRIIO/Mooncake(×2)/FlexKV/HF3FS/DecodeBench/Example |
| `vllm/distributed/ec_transfer/` | encoder cache 的姊妹框架（E/PD） |
| `vllm/v1/simple_kv_offload/`、`vllm/v1/hisparse/`、`vllm/v1/kv_hints/` | 极简 offload / 稀疏 / 编排提示 |

## 8. 配置面（速查）

- `vllm/config/cache.py`：block_size(73)/kv_cache_layout(80)/prefix_match_unit(91)/cache_dtype(123)/
  enable_prefix_caching(142)/hash_algo(144)/mamba_*(178-199)/num_gpu_blocks(220)/
  kv_offloading_*(256,262)/swa_bounded_replay(242, 需 MRV2)
- `vllm/config/scheduler.py`：watermark(197)/disable_hybrid_kv_cache_manager(183)/async_scheduling(209)
- `vllm/config/kv_transfer.py`：kv_connector/kv_role/engine_id/kv_load_failure_policy
- `vllm/config/kv_events.py`：publisher(zmq)/endpoint/buffer_steps

## 9. 测试地图（改哪层跑哪套）

| 层 | 测试 |
|---|---|
| core | tests/v1/core/（test_scheduler 7232 行、test_prefix_caching 5970、test_kv_cache_utils 4547、test_single_type、test_deferred_block_free、prefix_cache/ 子目录…） |
| worker | tests/v1/worker/（block_table、input_batch、model_runner×2） |
| backend | tests/v1/attention/（47 文件：parity、selection、layout、metadata builder） |
| kernel | tests/kernels/（test_cache_kernels、attention/ parity、mamba/） |
| connector | tests/v1/kv_connector/unit/（~65 文件：lifecycle、PD、错误传播、HMA、各 connector）+ nixl_integration/ |
| offload | tests/v1/kv_offload/（cpu/、tiering/） |
| 微基准 | benchmarks/kernels/（paged_attention、reshape_and_cache_flash、nvfp4…） |

## 10. 演进点与热点（review 时先想到这些）

**演进**：KVCacheManager 四层拆分｜v0 CUDA PagedAttention 移除｜V1/MRV2 双 runner｜
KVCacheLayout 布局系统｜HMA 契约｜connector 只剩 V1｜cache_finished_request 删除｜
config 改名（cache_config.py→config/cache.py，kv_cache_dtype→cache_dtype）

**热点（改这层必查）**：
1. 三套 block size（scheduler=LCM / hash=GCD·prefix_match_unit / group=per-spec·DCP）——多数 hybrid bug 在这
2. hybrid find_longest_cache_hit 固定点（coordinator:833）——最难 100 行
3. two-phase computed-block adoption（#33775）与 CoW 保留契约（_apply_cow→take_kv_cache_block_copies→_free_cow_retained_blocks）
4. deferred free fence 与 num_in_flight_tokens fence——free 错基础会 use-after-free/死锁
5. null block 永不 free/hash；free 顺序（hashed→tail LRU，unhashed→head LIFO）
6. get_num_blocks_to_allocate 与 allocate_new_blocks 必须镜像
7. EAGLE/MTP drop 贯穿查找-缓存-切块三处
8. 布局 stride 派生（create_kv_cache_views）——错则静默
9. CUDA graph 地址稳定与 PAD 不变量
10. 量化 scale 的三种粒度 dispatch
