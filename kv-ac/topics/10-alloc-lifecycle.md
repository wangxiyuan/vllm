# 10 · 分配生命周期：alloc/CoW/free/preempt（W2-D11/D12/D13）

## 核心问题
1. allocate_slots 全流程与闸门顺序？
2. CoW 的端到端契约？
3. free 顺序 / deferred free / fence？
4. preemption 与恢复？

## 代码索引
- `vllm/v1/core/kv_cache_manager.py` allocate_slots(371)/free(610)/evict_blocks(655)
- `vllm/v1/core/single_type_kv_cache_manager.py` get_num_blocks_to_allocate(178)/_apply_cow(449)
- `vllm/v1/core/sched/scheduler.py` preempt 循环(753-809)/preempt(1539)/_free_request(2628)/
  deferred(2679/2701/1986)/CoW 释放(1436/2690)/MRV2 分支(332)

## 种子结论
- allocate_slots 顺序：remove_skipped_blocks（先回收再计数）→ watermark 闸门（仅
  WAITING/PREEMPTED 且已有同批请求时）→ full_sequence_must_fit（可选整体准入）→
  get_num_blocks_to_allocate（=新块+可逐出的命中块+CoW 预留）vs free−reserved →
  two-phase adoption（先 touch 本地命中 #33775，再 external）→ allocate_new_blocks
  （含 CoW 重定向）→ cache_blocks（只缓存 verified tokens）。
- 镜像约束：get_num_blocks_to_allocate 与 allocate_new_blocks 必须逐分支一致，否则
  mid-prefill OOM/死锁（注释引 #33775/#39734）。
- CoW：partial hit 共享尾块 → _apply_cow 给本请求换私有新块，原块与新块端点进
  _pending_cow_copies → worker 执行 copy_kv_blocks → scheduler take_kv_cache_block_copies
  收副本（scheduler.py:1436）→ 完成后 _free_cow_retained_blocks(2690) 释放保留端。
  保留期内两块 hash 映射并存（BlockHashToBlockMap dict 分支）。
- free：finish→_free_request（connector 可 delay_free）→ 立即 free 或 deferred_frees
  （fence: last_sched_seq/processed_step_seq，update_from_output 时排水）。顺序必须
  reverse allocation order。
- preemption：recompute 模式——free 全部块、num_computed_tokens=0、重回 waiting；
  恢复时走 prefix 命中或重算。SWA 窗口回收受 num_in_flight_tokens fence（在飞 step
  还在读的块不能回收）。
- AsyncScheduler：每 output step cache_blocks（placeholder 计入），preempt 时清 stale 输出。

## 沉淀区
