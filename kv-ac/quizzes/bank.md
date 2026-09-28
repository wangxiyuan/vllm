# 自测题库（84 题，分级）

> 难度：L1 概念 · L2 代码 · L3 设计/推理 · L4 实战。
> 用法：①每日卡片完成后自测对应周的题；②`/kv-quiz weekN` 由导师抽 15 题判卷。
> 先遮住每周末尾的「参考要点」作答。错题自动回流 progress.md 薄弱点清单。

---

## W1 · 基础与全链路（20 题）

1. (L1) KV cache 里存的是什么？为什么 decoder-only 自回归推理必须有它？
2. (L1) 推导每层每 token 的 KV 字节数公式；GQA 相比 MHA 少在哪一项？
3. (L1) PagedAttention 三个核心机制（分页/共享/CoW）各解决什么浪费？
4. (L2) 默写本仓库 KVCacheSpec 继承树，并指出 MambaSpec、EncoderOnlyAttentionSpec 基类的特殊性。
5. (L2) 多个 group 的 block_size 不同时，scheduler_block_size 与 hash_block_size 分别怎么算？
6. (L2) create_kv_cache_views 的 C 维为什么以字节为单位？MHA 的 K/V 怎么共存于一个视图？
7. (L2) profile_run 做什么？为什么不用静态公式估显存？
8. (L2) available KV 显存从 profile 后的空闲显存扣掉哪些项？cudagraph 的份额在哪估？
9. (L2) num_blocks 为什么跨 TP/PP rank 取 min？
10. (L3) 为什么用单个 flat int8 buffer + as_strided 视图，而不是每层独立 tensor？至少 3 个理由。
11. (L3) KVCacheLayout 的 [L,B,H,N,C] 五维各是什么？LBHNC 与 LBNHC 分别对应 HND 还是 NHD？
12. (L3) manager block 256 拆成 kernel block 64 时，slot 与 block_table 的换算链怎么变？为什么要求 dense unpadded pages？
13. (L2) PAD_SLOT_ID 是多少？什么场景出现？kernel 怎么处理它？
14. (L2) 在 reshape_and_cache_flash_kernel 里指出 slot→(block,offset) 的换算行与 fp8 量化分支。
15. (L2) SM100 非 MLA 的默认 backend 是谁？SM90 呢？为什么 --block-size 会让 backend 静默降级？
16. (L3) FlashInfer 的 TRT-LLM decode 路径为什么强制 HND？物理布局转换发生在哪一行？
17. (L2) page table（block_table_tensor）与 slot_mapping 分别被写路径还是读路径消费？
18. (L4) TP=2 时 KV head 怎么切分？两个 rank 上的 KVCacheTensor shape 相同吗？
19. (L4) 手算：32 层、GQA 32q/8kv、head_size=128、bf16 的模型，每 token KV 多少字节？80G 卡 70% 可用、block=16 时最多同时服务多少 4K 序列（只算 KV）？
20. (L3) EncoderOnlyAttentionSpec 0 字节，为什么还要进 KVCacheConfig？

## W2 · 块管理核心（24 题）

21. (L1) prefix caching 的 hash 链为什么包含 parent hash？带来什么性质？
22. (L1) BlockHashWithGroupId 为什么要打包 group id？
23. (L2) free 时 hashed 块进 tail（LRU）、unhashed 块进 head（LIFO），各自优化什么？
24. (L2) BlockPool.touch 做哪两件事？为什么需要它？
25. (L2) null block 有哪些约束？哪些位置会指向它？
26. (L2) get_num_blocks_to_allocate 与 allocate_new_blocks 漂移会怎样？注释引用了什么问题？
27. (L2) two-phase computed-block adoption（#33775）的两个 phase 是什么？顺序错了会怎样？
28. (L3) 描述 CoW 的端到端契约：从 _apply_cow 到 _free_cow_retained_blocks，中间谁拷贝、谁保留、何时释放？
29. (L3) partial hash hit（Mamba align，block_size>hash_block_size）为什么必须用 CoW 保护共享尾块？
30. (L2) extra_keys 有哪四类？cache_salt 为什么只放进第一块的 hash？
31. (L2) NONE_HASH 的种子语义：sha256 系与 xxhash 系差异？对跨实例事件重建的影响？
32. (L2) prefix hit 查找为什么只在 waiting 请求上做？上限为什么是 num_tokens-1？
33. (L3) hybrid find_longest_cache_hit 的固定点迭代：举一个需要多轮收敛的例子。"full attention downward-closed" 在代码里的具体用途？
34. (L3) deferred free 的 fence（last_sched_seq/processed_step_seq）防什么竞态？构造一个不延迟就出错的序列。
35. (L2) num_in_flight_tokens 如何 fence SWA 窗口块回收？违反会发生什么？
36. (L2) 请求被 preempt 后恢复的两条路径是什么？
37. (L2) watermark 闸门只对哪类请求生效？为什么 running 请求不需要？
38. (L2) free_blocks 对入参顺序的要求是什么？违反的后果？
39. (L3) reset_prefix_cache 何时会失败？
40. (L2) BlockStored 事件的 parent_hash 让外部消费者能做什么？
41. (L2) PrefixCacheStats 在哪两个时机被记录？hit rate 从哪出去？
42. (L4) 两请求共享前缀至第 10 块，其中之一继续 decode——精确说出 CoW 触发时刻与两块的 hash/ref_cnt 状态变化。
43. (L3) AsyncScheduler 为什么每个 output step 都要 cache_blocks？num_output_placeholders 影响什么？
44. (L3) 设计题：给 prefix cache 加 TTL（N 秒不命中即提前淘汰），你会动哪些函数？哪些不变量不能破坏？

## W3 · Worker 与 Kernel（22 题）

45. (L1) V1 与 V2(MRV2) model runner 的定位差异？如何切换？哪些特性强制 V2？
46. (L2) Attention 层如何把 KVCacheSpec 报告给 runner（注册链）？
47. (L2) 一个 Mamba 页在 bind 阶段怎么变成 conv_state 与 ssm_state 两个视图？
48. (L2) mamba_cache_mode 的 none/all/align 各是什么语义？align 为什么引入 partial hash？
49. (L2) fp8 KV 的 k_scale/v_scale 两条流向（写侧/读侧）各在哪个函数/参数？
50. (L3) per-tensor、per-head、per-token-head 三种量化粒度的 dispatch 条件（kv_scale_stride 分支）？
51. (L2) fp8_e5m2 与 fp8 checkpoint 的冲突检查在哪？为什么会有冲突？
52. (L2) nvfp4 与 fp8 的块结构差异？scale 怎么组织？
53. (L2) AttentionCGSupport 四级各自约束？UNIFORM_BATCH 什么意思？
54. (L3) CUDA graph 回放为什么要求「地址稳定」？哪些张量受此约束、谁保证？
55. (L2) build_for_drafting 与普通 build 的输入差异？
56. (L3) 被拒绝的 spec token 为什么不需要清理已写的 KV？两侧分别谁把 num_computed_tokens 减回去？
57. (L3) Mamba 为什么必须显式 checkpoint/rollback？checkpoint 在什么时机做？
58. (L2) triton_unified_attention 为什么说 layout-agnostic？它的 cache stride 参数从哪来？
59. (L2) v0 CUDA PagedAttention kernel 在本仓库的现状？哪些平台还在用 paged_attention 路径？
60. (L3) MLA cache 为什么是 [B, block, 1, kv_lora_rank+qk_rope_dim] 单张量？以 DeepSeek 参数估算相对 MHA 的压缩比？
61. (L2) concat_and_cache_mla 与 reshape_and_cache_flash 的输入差异？
62. (L2) SM100 上 MLA backend 的优先级链？fp8 KV 为什么偏向 FlashInfer sparse？
63. (L4) 实验：同一模型切 flashinfer/flash_attn/triton，记录吞吐差异并从 kernel 特性解释。
64. (L4) 给 reshape_and_cache_flash 的 per-head scale 路径写一个正确性单测（参考 test_cache_kernels.py 结构）。
65. (L3) kernel block 切分要求 dense unpadded pages 的原因？报错时建议的 VLLM_KV_CACHE_LAYOUT=LBNHC 为什么能救？
66. (L2) draft 与 target 的 KV 在同一个 pool 吗？is_eagle_group 怎么影响缓存行为？

## W4 · 分布式与实战（18 题）

67. (L1) KV connector 解决哪三类问题？举对应 connector。
68. (L2) SCHEDULER/WORKER 两个 role 的实例分别挂在哪个对象、由谁创建？
69. (L2) get_num_new_matched_tokens 的三种返回值语义与调度后果？
70. (L2) build_connector_meta 的产物怎么到达 worker、何时 bind/清除？
71. (L2) request_finished 返回 (delay_free, kv_transfer_params)——delay_free 改变了块生命周期的什么？
72. (L3) pull 与 push 模式下块释放安全性差异？lease/heartbeat 防什么？
73. (L3) 列出一次 PD 请求（P prefill→D decode）≥8 个钩子调用点（按时序）。
74. (L2) OffloadingConnector 的 canonical_mapping 为什么是唯一感知 TP/DCP/PCP 的模块？"canonical pages" 是什么抽象？
75. (L2) MultiConnector 的 load/save 组合语义？失败回退？
76. (L2) HMA（SupportsHMA）契约为什么出现？request_finished_all_groups 语义？
77. (L2) failed_recving 后 kv_load_failure_policy 的 recompute|fail 各发生什么？invalid_block_ids 防什么污染？
78. (L2) BlockStored 事件怎么配置外发（ZMQ）？llm-d/production-stack 这类路由器拿它做什么？
79. (L3) ec_transfer 为什么不直接复用 kv_transfer 框架（encoder cache 的什么差异）？
80. (L4) 用 ExampleConnector 写 mini 文件 connector：列出必须实现的钩子与 metadata 设计。
81. (L4) 按 AGENTS.md：一个 kv cache bugfix PR 必须包含哪些检查、声明与测试证据？
82. (L3) review 题：某 connector 在 request_finished 里直接改 BlockPool 的块状态——指出违反的边界与正确姿势。
83. (L3) 设计题：新增"跨实例去重共享存储 connector"，列出可复用的框架设施与要新写的部分。
84. (L2) kv_load_failure_policy 与 admission_control 的职责边界？

---

## 参考要点（作答后对照；详情以源码为准）

### W1
1. 每层注意力层的 K/V 投影缓存；自回归时历史 token 的 K/V 不变，避免重复计算。
2. 2×kv_heads×head_size×dtype_bytes；GQA 少在 kv_heads（多 q 共享一组 KV）。
3. 分页→外部碎片；共享→相同前缀物理复用（prefix caching）；CoW→共享块被追加时才复制，兼顾共享与写安全。
4. 见 topics/02；MambaSpec 直继 KVCacheSpec（非 Attention 语义）；EncoderOnly 0 字节。
5. scheduler_block_size=各组 LCM(kv_cache_utils:705)；hash_block_size=prefix_match_unit 或 GCD。
6. 字节为单位让不同 dtype/layout 统一寻址；MHA 的 K/V 平分 C（C=2×head_size）再 view 拆分。
7. 真·dummy 前向测 activation 峰值，含 compile/后端 workspace 等动态开销。
8. 非 KV 常驻（权重已扣）、activation 峰值（已反映）、cudagraph 预留、确定性开销；CG 份额在 gpu_worker 653-660 估算。
9. 块编号要全局一致，任一 rank 不足都会越界。
10. ①一次分配减碎片/分配器压力②统一逻辑形状任意视图③块 id 偏移可移植（offload/跨实例）④整体 zeroing/镜像方便。
11. L=layer,B=block,H=head,N=token(slot),C=channel(bytes)；LBHNC 头在外≈HND，LBNHC token 在外≈NHD。
12. page_table 编号变 kernel block 粒度（blocks_per_kv_block 映射 b*bpk+k）；切分要求物理页内连续无 padding，否则视图 stride 无法表达。
13. -1；CUDA graph pad 的假 token / draft pad；kernel guard 跳过写入。
14. cache_kernels.cu:326 起：slot_mapping[t]→block=slot/bs、off=slot%bs；量化按 kv_scale_stride==0 走 per-tensor fast path。
15. SM100→FLASHINFER；SM90→FLASH_ATTN；block-size 与 backend 约束冲突时回退选择并 warn（cuda.py:507-524）。
16. trtllm-gen kernel 的页布局按 HND 优化；permute 在 flashinfer.py:2160。
17. slot_mapping→写；block_table_tensor→读。
18. kv_heads 按 TP 均分（不能整除时复制/对齐）；shape 的 H 维减半，num_blocks 相同。
19. 每 token=2×8×128×2=4KB；70%×80G=56G→14.7M token；4K 序列→约 3600 个并发（上限还会受其他开销压缩）。
20. 维持 layer→group 完整映射与注册校验（KVCacheSpecKind 消费），不占显存但参与一致性检查。

### W2
21. 链式使同前缀必同 hash，一次查表确认整段；防局部碰撞导致的假命中。
22. 不同 group 物理页大小/管理器不同，同一逻辑前缀的"可复用性"按组隔离。
23. tail=最后复用（LRU 保命中）；head=最先复用刚释放页（LIFO 保 GPU cache 局部性）。
24. ref_cnt++ 并从 free 队列摘除；命中块被新请求引用时必须脱离可逐出状态。
25. ref_cnt 不维护、永不 free、永不 hash；SWA/ChunkedLocal 的窗口外位置、skip 位置用其占位。
26. 中途 OOM/死锁（注释引 #33775/#39734）：预估少了→分配时才发现，陷入 preempt 风暴；预估多了→水位失真。
27. phase1 touch 本地命中块；phase2 分配 external（connector）块。反了会在 external 分配失败时把已 touch 的命中块留在中间态（池状态与请求不一致）。
28. _apply_cow 换私有块并登记两端→scheduler take_kv_cache_block_copies(1436) 取走副本请求→worker copy_kv_blocks 执行→完成后 _free_cow_retained_blocks(2690) 释放保留端；期间同 hash 双块（dict 分支）。
29. 尾块含未定界 token，直接共享会让后续追加破坏他人前缀语义；CoW 延迟到边界确定。
30. LoRA/MM(identifier+offset)/cache_salt/prompt-embeds SHA；salt 是租户隔离，只需区分入口。
31. sha256 系确定性种子→跨进程/节点可复现（事件可重建）；xxhash 默认随机→同进程内一致性，跨实例不保证。
32. running 请求的前缀已在计算中；留 1 token 保证至少现算一步以生成首个新 KV。
33. 例：SWA 组命中受限于窗口内块是否都被 full 组覆盖；downward-closed 用于单调裁剪（命中 k⇒前缀 k-1 皆命中）。
34. 块 free 后立即被下一请求 get_new_blocks 拿走并写入，而上一步的 worker 还在读——fence 保证排水发生在 processed_step_seq 追平之后。
35. 在飞 step 可能仍在读即将滚出窗口的块；立即回收=use-after-free（见 test_swa_inflight_window_free）。
36. 重算（recompute，默认）或依赖 prefix cache 重新命中已缓存前缀。
37. WAITING/PREEMPTED 且本步已有其他请求被调度；running 请求每步只增小块数，须尽量不死锁。
38. 必须按淘汰优先序（reverse allocation order）；错序会破坏 LRU/LIFO 语义，命中率静默劣化。
39. 还有非 null 块被引用（含在飞引用）时失败，返回 False 由调用方决定是否 preempt。
40. 重建前缀树/块链，实现 KV 感知路由与跨实例预热。
41. schedule 内 record_prefix_cache_stats(1259) 与 make_stats(2852) 重置返回；最终进 SchedulerStats.prefix_cache_stats。
42. 触发点：继续 decode 的请求首次 allocate_slots 发现共享尾块为 partial 命中→_apply_cow 换私有块；原块 hash/ref_cnt 不变（服务另一请求），新块无 hash、ref_cnt=1；两端进 _pending_cow_copies 至 worker 拷贝完成释放保留端。
43. async 输出占位使"已产出 token"分步到达，逐步缓存才能保住命中链；placeholder 影响缓存边界与 stale 回滚。
44. 思路：BlockPool 记 last_hit_ts（touch 时更新）+ 周期清扫或惰性淘汰；不得破坏 free 顺序不变量与 hash 一致性（清 hash 发 BlockRemoved、保留 reachable 不变量）。

### W3
45. V1 巨石兼容一切；V2 重构执行主路径（更小、可维护、支持 PCP/ubatch 等新特性）；use_v2_model_runner 切换；HiSparse 与 swa_bounded_replay 强制 V2。
46. Attention.__init__ 注册 static_forward_context → runner get_kv_cache_spec(7421) 遍历收集 → 分组。
47. bind_kv_cache 对 MambaSpec 按 shapes/dtypes 把页的字节 C 维切片成 conv/ssm 两个连续视图（mamba/abstract:29-44）。
48. none=不缓存可复用状态语义；all=整块缓存；align=放大页对齐 hash 边界并启用 partial hash+checkpoint。
49. 写侧：reshape_and_cache_flash(...,k_scale,v_scale)（cache_kernels.cu 分支）；读侧：k_descale/v_descale 进 backend forward。
50. kv_scale_stride==0→per-tensor；>0→per-head（[num_heads]）；per-token-head 是独立 KVQuantMode 走专 kernel。
51. attention.py:221-234；e5m2 无 scale 语义与 checkpoint 的 per-tensor scale 方案冲突，需 quant 方法声明无 kv_cache_scheme。
52. nvfp4 每 block 存 scale 因子（微块浮点），有专门 kernel 族与 4over6/DS-MLA 变体。
53. ALWAYS 任意批可图；UNIFORM_BATCH 同构批；UNIFORM_SINGLE_TOKEN_DECODE 仅 decode；NEVER 走 piecewise。
54. 捕获期指针被烧进 graph；page table/slot/persistent batch 必须原地更新；block_table.py:182-238 注释与 cudagraph_utils 保障。
55. draft 需要按 draft token 树构建的 qox/页表（多层 lookahead），非单步连续。
56. KV 槽位语义是"下次写入覆盖"，无需清理；scheduler update_from_output(2057-2066) 与 worker(5127-5225) 各自回滚计数。
57. 循环状态覆写不可逆；prefill 结束/对齐边界做 checkpoint（MambaManager 1675-2031），拒绝时回滚。
58. 它只依赖传入的 stride/页表而非固定布局常量（triton_attn.py:395-421）；stride 来自 create_kv_cache_views。
59. 已删除（仅 ops/paged_attn.py wrapper）；CPU/XPU/ROCm 仍走 paged_attention(_custom_ops.py:104)。
60. cache 只存压缩 latent+rope（kv_lora_rank+rope_dim），MHA 存 2×n_kv_heads×head_size；DS 类参数下约减一个数量级（按具体 kv_lora_rank 算）。
61. MLA 输入是 kv 投影直接 concat 进 latent 页；MHA 是 K/V 各自 reshape 进分视图。
62. FLASHINFER_MLA→TOKENSPEED_MLA→CUTLASS_MLA→FLASH_ATTN_MLA→FLASHMLA→TRITON_MLA→sparse…；fp8 偏 FlashInfer sparse 因其支持 fp8 页读与 indexer。
63. 记录数值即可；解释点：FA 变长批友好、FlashInfer SM100 tensor core 优化、Triton 灵活但常量开销大。
64. 参照 test_cache_kernels.py：构造 [H] scale 输入，断言写入值=quant(k)*scale 且与 per-tensor 路径一致性。
65. 切分要求块内页连续可整除，padding 页破坏 stride 数学；LBNHC 把 token 维放外层使子块视图连续。
66. 同一 pool（eagle group 是 config 里的组）；is_eagle_group 组多缓存一块（lookahead）且查找末块 drop。

### W4
67. PD 分离（Nixl/Mooncake）、分层 offload（OffloadingConnector/MooncakeStore/FlexKV）、跨实例共享/去重（MooncakeStore/LMCache）。
68. scheduler 持有 SCHEDULER role（scheduler.py:165 创建）；worker 侧单例 _KV_CONNECTOR_AGENT（kv_transfer_state）持 WORKER role。
69. (None,_)：本轮推迟调度；(n,False)：同步加载 n token（当步前完成）；(n,True)：异步加载（先调度后到齐再 forward）。
70. scheduler_output.kv_connector_metadata 携带→runner execute_model 前 bind_connector_metadata→步末 clear；W→S 用 worker meta aggregate。
71. 块在 connector 确认发送/接收完成前保持被引用（不进 free 池）；完成信号经 get_transfer_results/finished_sending 回流后才释放。
72. pull：D 读 P，P 侧块生命周期由 finished_sending+lease 续约保护；push：P 写 D 预分配块，需 D 侧确认；heartbeat 防对端失联后块被永久占用（#41383 lease）。
73. 示例序：P request_finished→delay_free→finished_sending；D get_num_new_matched_tokens→allocate→update_state_after_alloc→build_connector_meta→start_load_kv→wait_for_layer_load(逐层)→wait_for_save→get_transfer_results→update_connector_output。
74. offload 目标机器的并行布局可能与本机不同，canonical pages 把逻辑页映射成与并行无关的规范序以便迁移；它是唯一需要理解 TP/DCP/PCP 切分的地方。
75. load 取配置序中第一个广告命中的；save 广播到所有；HMA 支持取所有子 connector 的交集。
76. hybrid 多 group 下"请求结束"需按组全量确认释放（request_finished_all_groups），否则逐组 request_finished 语义在混合管理器下有歧义；factory 强制校验。
77. recompute=作废 invalid 块并回退重算（evict 防脏读）；fail=请求失败上报；invalid_block_ids 防把传输出错/不完整的块当有效前缀（cache pollution）。
78. config/kv_events.py 开 zmq publisher→endpoint；路由器用 parent_hash 重建前缀树做 KV-aware 负载均衡与预热。
79. encoder cache 生命周期短、非分页、以多模态项为粒度、常走不同介质（/dev/shm），复用会引入错误抽象；但框架形态（S/W role+metadata+events）一致。
80. 见 curriculum/week4.md D29 六钩子清单；metadata=块 hash→文件路径表。
81. gh 重复性检查；pre-commit；相关 pytest 全绿；影响输出时模型 eval；PR 描述含动机/测试证据/AI 声明；human submitter 署名跑测。
82. 违反：connector 不拥有 pool 内部状态，只能通过返回值（delay_free/params）与专用钩子（evict_blocks、take_events）影响池；正确姿势见 OffloadingConnector/Nixl 对 request_finished 的用法。
83. 可复用：factory 注册、KVConnectorMetadata 通道、BlockStored 事件流、handshake/stateless_coordinator、OffloadingConnector 的 canonical mapping；需新写：远端索引协议与去重键（hash 已全局可用）。
84. 前者是 KV 传输失败时的块级恢复（recompute|fail）；后者是 API server 侧的请求准入控制，不含 KV 语义。
