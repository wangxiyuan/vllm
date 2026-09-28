# Week 2（D8-D14）块管理核心（模块心脏）

> 周目标：徒手模拟 10 步 alloc/free/CoW 序列画出 pool 状态；新增 1 个测试 case 通过；
> tests/v1/core 全绿。这是社区 review 流量最大的一层。对应讲义：topics/08~11。

> 提醒：本层"坑"密度最高（free 顺序、CoW 保留契约、fence、null block、三套 block size），
> 遇到注释里引用 issue 号（#33775、#39734 等）一定点进去看背景。

---

## D8 核心数据结构

**代码索引**
- `vllm/v1/core/block_pool.py`：BlockPool(135)、BlockHashToBlockMap(34)（单块 vs dict 分支）、
  null_block 创建(186)、get_cached_block(197)、cache_full_blocks(225)、_build_block_stored_event(345)、
  emit_cached_block_events(375)、cache_partial_block(448)、_remove_cached_block_hashes(592)、
  _insert_block_hash(628)、move_block_hashes(650)、get_new_blocks(668)、_maybe_evict_cached_block(731)、
  touch(754)、free_blocks(776)、evict_blocks(809)、reset_prefix_cache(829)、get_usage(879)、take_events(892)
- `vllm/v1/core/kv_cache_utils.py`：KVCacheBlock(176, slots dataclass)、set_block_hash/reset_hash(211/222)、
  FreeKVCacheBlockQueue(247)（侵入式双向链表、fake head/tail、O(1) 中间摘除）

**驱动问题**
1. KVCacheBlock 为什么用 `__slots__`？pool 字段（多池延迟释放）做什么用？
2. FreeKVCacheBlockQueue 为什么不用 `collections.deque`？（中间 O(1) 摘除 + 顺序语义）
3. BlockHashToBlockMap 什么时候从"单块"退化为 dict？同一 hash 多块从哪来（CoW 保留、partial hit）？
4. touch 除了 ref_cnt++ 还做什么？
5. null block 的所有约束在哪体现？谁指向它、为什么绝不能 free/hash？

**实验**：跑 tests/v1/core/test_kv_cache_utils.py 中 free-queue 相关 case；自己写 5 行脚本演示 popleft/remove 顺序

---

## D9 hash 链与多粒度

**代码索引**
- `vllm/v1/core/kv_cache_utils.py`：BlockHash(62)、BlockHashWithGroupId(75-94)、ExternalBlockHash(72)、
  init_none_hash(161)、generate_block_hash_extra_keys(611)、hash_block_tokens(650)、
  resolve_kv_cache_block_sizes(705)、BlockHashListWithBlockSize(2781)、resolve_block_hashes(2857)
- `vllm/v1/request.py`：block_hashes、update_block_hashes(225/272/285)
- `vllm/v1/engine/core.py`：init_none_hash + get_request_block_hasher(230-236)

**驱动问题**
1. hash 为什么包含 parent_hash（链式）？两次相同前缀的请求 hash 链如何复用？
2. BlockHashWithGroupId 打包 group id 进 bytes 解决什么？ExternalBlockHash 给谁用？
3. extra_keys 四类（LoRA/MM/cache_salt/prompt-embeds）各自防什么误命中？cache_salt 为什么只放第一块？
4. NONE_HASH 种子：sha256 为什么用确定性种子、xxhash 为什么默认随机？对跨实例事件复现什么影响？
5. prefix_match_unit / hash_block_size / scheduler_block_size 各自怎么算（GCD/LCM/DCP）？
6. 请求侧 hash 链在什么时候增量更新（append_output_token_ids）？

**实验**：跑 test_kv_cache_utils.py 中 hash 相关 case；构造两个 prompt 验证前 N 块 hash 相同

---

## D10 查找：get_computed_blocks 与固定点

**代码索引**
- `vllm/v1/core/kv_cache_manager.py`：get_computed_blocks(264)（skip 条件、num_tokens-1 截断、
  full 模式重发事件 305-310）、get_computed_blocks_for_connector(323)、truncate_computed_blocks(820)
- `vllm/v1/core/kv_cache_coordinator.py`：三协调器(471/522/607)、verify_and_split_kv_cache_groups(728)、
  **find_longest_cache_hit 固定点(833)**（全模块最难的 100 行）、find_longest_cache_hit_per_group(970)、
  enable_partial_hash_hits(693)、num_uncached_common_prefix_tokens / shared_prefix_boundary
- `vllm/v1/core/single_type_kv_cache_manager.py`：FullAttentionManager.find_longest_cache_hit(742，
  链式扫描 792、细粒度内部探测 802、EAGLE drop 829)、reachable_block_mask(530)

**驱动问题**
1. 为什么只在 waiting（num_computed_tokens==0）请求上做查找？为什么上限是 num_tokens-1？
2. 固定点迭代为什么会发生（举例：SWA 组命中长度依赖 full 组命中长度的回调）？
3. "full attention is downward-closed" 在这段代码里的具体用法（trimming）？
4. partial hash hit 的内部探测：block_size > hash_block_size 时 probe 什么？命中边界怎么对齐？
5. hybrid 命中分歧（divergent hits）什么时候出现？connector 怎么补齐？
6. reachable_block_mask 保证什么不变量？

**实验**：对 tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py 逐 case 断点理解（加 print）

---

## D11 分配：allocate_slots 全流程 + CoW

**代码索引**
- `vllm/v1/core/kv_cache_manager.py`：allocate_slots(371)——watermark 闸门(506-513)、
  full_sequence_must_fit(515-531)、remove_skipped_blocks 先行(547)、get_num_blocks_to_allocate(553)、
  容量检查(566-570)、allocate_new_computed_blocks(578，two-phase touch，#33775)、
  allocate_new_blocks(585)、cache_blocks(606)、take_kv_cache_block_copies(888)
- `vllm/v1/core/single_type_kv_cache_manager.py`：get_num_blocks_to_allocate(178，admission cap +
  CoW +1 预留)、add_local_computed_blocks(265)、allocate_external_computed_blocks(324)、
  allocate_new_blocks(364，CoW 重定向)、_apply_cow(449)、cache_blocks(471)、_pending_cow_copies(150)
- `vllm/v1/core/sched/scheduler.py`：preemption 循环(753-809)、take_kv_cache_block_copies 调用(1436)、
  _free_cow_retained_blocks(2690)

**驱动问题**
1. watermark 为什么只对 WAITING/PREEMPTED 请求生效？running 请求小步分配为什么不需要？
2. get_num_blocks_to_allocate 与 allocate_new_blocks 为什么必须严格镜像？漂移的后果（注释引用了什么 issue）？
3. two-phase adoption：touch 本地命中 与 分配 external 块，为什么顺序不能反（#33775）？
4. CoW 全链路：_apply_cow 换了什么 → _pending_cow_copies 里存了什么 → 谁真正执行拷贝（worker 侧
   copy_kv_blocks）→ retained 块何时释放？
5. schedule() 主循环里分配失败后的 preempt 循环怎么工作（最多重试几次、free 掉什么）？
6. cache_blocks 为什么只缓存 verified tokens？num_reprefillable_tokens（MTP 多模块）怎么跳过？

**实验**：跑 tests/v1/core/test_prefix_caching.py -q；构造两请求共享前缀的日志复现 CoW

---

## D12 释放、淘汰与窗口回收

**代码索引**
- `vllm/v1/core/kv_cache_manager.py`：free(610)、pop_blocks_for_free(641)、remove_skipped_blocks(621)、
  evict_blocks(655)、reset_prefix_cache(664)
- `vllm/v1/core/block_pool.py`：free_blocks(776)（**hashed→tail LRU / unhashed→head LIFO**）、
  get_new_blocks(668) + _maybe_evict_cached_block(731)（腾位时清 hash + BlockRemoved 事件）、
  evict_blocks(809)、reset_prefix_cache(829)
- `vllm/v1/core/sched/scheduler.py`：finish_requests→_free_request(2628)、_free_request_blocks(2679)、
  deferred frees 与 _drain_deferred_frees(2701/1986)、fence 字段 last_sched_seq/processed_step_seq
- `vllm/v1/core/single_type_kv_cache_manager.py`：remove_skipped_blocks——SWA(1148 get_num_skipped_tokens)、
  RSWA(913 gap 回收)、ChunkedLocal(1392)、_remove_blocks_in_range(655)（换 null_block）
- 测试：tests/v1/core/test_deferred_block_free.py、test_swa_inflight_window_free.py

**驱动问题**
1. free 顺序语义：为什么要求按淘汰优先序（reverse allocation order）？搞错会静默发生什么？
2. hashed 块进 tail、unhashed 块进 head——各自优化什么（命中率 vs GPU 局部性）？
3. deferred free 防什么竞态？举一个「不延迟就会 use-after-free」的坏序列？
4. num_in_flight_tokens 是什么？它如何 fence 掉「还被在飞 step 读着的窗口块」的回收？
5. RSWA 的 gap 回收回收的是哪些块？与 SWA 的窗口滚动有何不同？
6. evict_blocks（连接器显式逐出）与自然淘汰的差异？_handle_invalid_blocks 何时调用它？
7. reset_prefix_cache 什么时候会失败？

**实验（破坏性，学完再拆）**：故意把 free_blocks 的 append 端对调 → 跑 test_prefix_caching 看命中率回归；
故意去掉 CoW 预留的 +1 → 观察 admission/分配错位（先 git stash 记得还原！）

---

## D13 调度集成与指标

**代码索引**
- `vllm/v1/core/sched/scheduler.py`：schedule() 主循环(557 起)、waiting loop 与 prefix lookup(932 附近)、
  record_prefix_cache_stats(1259)、preempt(1539)、update_from_output 的回滚(2057-2066)与事件排水(2303-2318)、
  finish/_free_request(2628)、reset(2757)、make_stats(2852/2858/2874)、MRV2 分支(332)
- `vllm/v1/core/kv_cache_manager.py`：usage(226)、make_prefix_cache_stats(236)、take_events(716)
- `vllm/v1/core/kv_cache_metrics.py`：KVCacheMetricsCollector(46)（~1% 采样）、KVCacheEvictionEvent
- `vllm/v1/core/sched/async_scheduler.py`：output placeholder 与每步 cache_blocks(69-74)

**驱动问题**
1. 一次 schedule() 从 waiting 队列到 SchedulerOutput 的完整步骤（≤12 步列出）？
2. PrefixCacheStats 在哪两个时机记录？hit rate 最终从哪个 stats 对象出去？
3. kv_cache_usage 指标怎么算（BlockPool.get_usage 879）？watermark 在监控上怎么体现？
4. async scheduling 下缓存时机为什么变了？num_output_placeholders 影响什么？
5. MRV2 下 scheduler 的分支（decode 节流节奏）差异？

**实验**：起服务压一轮多轮对话负载，记录 hit_rate 与 kv_cache_usage 曲线（metrics 端点）

---

## D14（10-11 周日）测试日 + 复盘

- 上午：tests/v1/core 全量跑绿（约 25 个文件）；挑 test_prefix_caching.py 里一个场景**新增一个 case**（如
  「三请求两两共享前缀 + 中途 preempt」）并跑通 —— 这就是 W2 里程碑的测试产出
- 下午：`/kv-quiz week2` 15 题 → 错题入薄弱点；`/kv-progress` 盘点
- 产出：徒手模拟的 pool 状态机图（拍照/存 topics/10 沉淀区）

**周验收**：里程碑 2 达成。
