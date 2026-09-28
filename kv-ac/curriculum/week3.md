# Week 3（D15-D21）Worker、Kernel 与高级特性

> 周目标：跨 backend 对比报告；能逐行讲 reshape_and_cache_flash_kernel；kernel 单测通过。
> 对应讲义：topics/12~13（+04/06/07 深化）。注意双代 runner：读代码前先确认读的是哪一代。

---

## D15 双代 Model Runner 对照

**代码索引**
- V1：`vllm/v1/worker/gpu_model_runner.py`（7528 行）——_update_states(1192)、zeroing 新块(1221)、
  _get_slot_mappings(4065)、execute_model(4149)、_update_states_after_model_execute(1564)、
  initialize_attn_backend(6966)、initialize_metadata_builders(7063)、may_reinitialize_input_batch(7171)、
  initialize_kv_cache_tensors(7260)、initialize_kv_cache(7332)、get_kv_cache_spec(7421)
- V2：`vllm/v1/worker/gpu/model_runner.py`（2345 行）+ gpu/ 包（block_table、input_batch、spec_decode/、
  cudagraph_utils）、`vllm/v1/worker/gpu/README`（若有）
- 选择器：`vllm/v1/worker/gpu_worker.py` 510-526；开关 `use_v2_model_runner`（config/vllm.py:701、
  config/compilation.py:1376、env VLLM_USE_V2_MODEL_RUNNER）；scheduler 分支(332)；xpu 的 V2 参照

**驱动问题**
1. V2 拆分的动机是什么（从 scheduler 的 MRV2 分支与 AsyncScheduler 的 next_decode_eligible_step 反推）？
2. 哪些特性被强制拉到 V2（HiSparse、swa_bounded_replay）？为什么？
3. 两代里同一个功能（slot mapping/block table/input batch）各在哪些文件？差异是什么？
4. zeroing 新块(1221) 为什么存在？get_zeroing_block_ids_in_range 谁调用？
5. ubatch wrapper（gpu_ubatch_wrapper.py）解决什么？与 connector finalize 时机怎么互相影响？

**实验**：同一模型分别用 V1/V2 起服务，对比启动日志与吞吐

---

## D16 Mamba / 混合模型 Worker 侧

**代码索引**
- `vllm/v1/kv_cache_interface.py`：MambaSpec(1032-1110)（shapes/dtypes、num_speculative_blocks、
  page_size_padded、mamba_cache_mode）
- `vllm/v1/worker/mamba_utils.py`（1687 行）：get_mamba_groups(689)、spec-decode GPU context(770)、
  状态拷贝/对齐 helper、Triton batch memcpy
- `vllm/model_executor/layers/mamba/abstract.py`：get_kv_cache_spec(57)、bind 拆页(29-44)
- mixer 层写状态：mamba_mixer2.py / gdn / kda、ops/ssu_dispatch.py（causal_conv1d_update、
  selective_state_update）
- 测试：test_single_type_kv_cache_manager.py（mamba 段）、test_mamba_align_chunk_split.py、
  tests/v1/e2e/general/test_mamba_prefix_cache.py

**驱动问题**
1. mamba_cache_mode 三种（none/all/align）语义？align 为什么把块变大并启用 partial hash？
2. 一个 mamba "页"怎么被 bind 拆成 conv_state 与 ssm_state 两个视图？
3. mamba 的 prefix caching 靠什么保证正确（状态不能部分共享时怎么办）？checkpoint 机制(1675-2031)做什么？
4. spec decode 下 mamba 为什么需要显式 checkpoint/rollback？num_speculative_blocks 是什么？
5. scheduler 侧 MambaManager(1443) 与 shared-prefix checkpoint(199 配置) 的关系？

**实验**：起一个 hybrid 模型（Qwen3-Next 类或 Jamba 类），观察其 KVCacheConfig 分组与 num_blocks

---

## D17 量化 KV Cache

**代码索引**
- `vllm/v1/kv_cache_interface.py`：KVQuantMode(39)
- `vllm/config/cache.py`：cache_dtype 值域(39-57)（fp8/fp8_e4m3/fp8_e5m2/nvfp4/nvfp4_4over6/
  nvfp4_ds_mla/int8/int4/fp8_per_token_head）
- `vllm/model_executor/layers/attention/attention.py`：_k_scale/_v_scale buffer(136-153)、
  量化方法注册(157)、fp8_e5m2 与 checkpoint 冲突(221-234)
- scale 两条流：写侧 `reshape_and_cache_flash(..., k_scale, v_scale)`（cache_kernels.cu 326 起）；
  读侧 k_descale/v_descale 进 backend kernel
- nvfp4：csrc libtorch_stable nvfp4_kv_cache_kernels.cu、benchmarks/kernels/benchmark_nvfp4_quant.py
- 文档：docs/features/quantization/quantized_kvcache.md

**驱动问题**
1. k_scale/v_scale 从哪来（checkpoint 的 BaseKVCacheMethod vs 静态 1.0）？per-head scale 的 shape 约定？
2. 写侧量化与读侧 descale 为什么两侧都要支持？e5m2 为什么不需要 scale？
3. per-tensor / per-head / per-token-head 的 dispatch 条件（kv_scale_stride 分支）？
4. nvfp4 与 fp8 的块结构差异？哪些 backend 支持 nvfp4 读？
5. 支持矩阵：哪些 backend 拒绝哪些 cache_dtype（supports_kv_cache_dtype）？

**实验**：fp8 e4m3 开关跑 accuracy 快测 + 吞吐对比；记录显存变化

---

## D18 CUDA Graph 与 Attention 元数据

**代码索引**
- `vllm/v1/attention/backend.py`：AttentionCGSupport(559)（ALWAYS/UNIFORM_BATCH/
  UNIFORM_SINGLE_TOKEN_DECODE/NEVER）、build_for_cudagraph_capture、update_block_table
- `vllm/v1/worker/gpu/cudagraph_utils.py`：CudaGraphManager(142)、batch descriptor
- `vllm/v1/worker/gpu/block_table.py`：地址稳定注释(182-238)、PAD(326-328)
- connector 交互：`requires_piecewise_for_cudagraph`（kv_connector/v1/base.py:658，LMCache layerwise）

**驱动问题**
1. 四级 CG support 各自的约束？为什么有的 backend 只支持 UNIFORM_SINGLE_TOKEN_DECODE？
2. 捕获期「地址稳定」不变量到底指什么（哪些张量、谁保证、注释在哪）？
3. 回放时 batch 变小怎么处理（pad/idx_mapping=-1/PAD_SLOT_ID）？
4. 什么时候必须退到 piecewise cudagraph（connector layerwise 等）？
5. build_for_cudagraph_capture 与普通 build 的输入差异？

**实验**：开/关 cudagraph 对比 decode 吞吐；观察 capture 日志里各 backend 的 CGSupport 等级

---

## D19 Kernel 精读日

**代码索引**
- `csrc/libtorch_stable/cache_kernels.cu`：逐行精读 reshape_and_cache_flash_kernel(326)——
  索引计算、向量化 copy、量化 scale 分支、HND path；concat_and_cache_mla_kernel(414/462/516)
- fused 变体：cache_kernels_fused.cu；nvfp4：nvfp4_kv_cache_kernels.cu
- Triton：`vllm/v1/attention/ops/triton_unified_attention.py` 全文（page table + stride 消费）、
  triton_reshape_and_cache_flash.py
- 历史定位：`vllm/v1/attention/ops/paged_attn.py`（51 行 wrapper；v0 CUDA kernel 已移除，
  CPU/XPU/ROCm 走 ops.paged_attention / paged_attention_rocm(_custom_ops.py:104)）
- manager→kernel block 切分链：prepare_kernel_block_sizes → BlockTables.blocks_per_kv_block →
  _compute_slot_mappings_kernel → append_block_ids（b*bpk+k 重映射）
- 微基准：benchmarks/kernels/benchmark_reshape_and_cache_flash.py、benchmark_paged_attention.py

**驱动问题**
1. kernel 里一次 copy 的向量宽度由什么决定？为什么 HND path 慢/特殊？
2. slot=-1 的 guard 在哪？block 越界会怎样？
3. triton_unified_attention 怎么用 page table 遍历 K/V？stride 参数从哪个 view 来？
4. 为什么本仓库删了 v0 CUDA PagedAttention（从调用方现状反推维护成本）？
5. manager block 切分成 kernel block 后，block_table 里的编号是哪一层的编号？

**实验**：跑 benchmark_reshape_and_cache_flash（CUDA vs Triton 对比记录数字）；kernel 单测全绿

---

## D20 MLA 专题

**代码索引**
- spec：MLAAttentionSpec(656，_apply_alignment_padding 646-654)、SlidingWindowMLASpec(922)
- 写入：backend.py MLA 基类(1037 起) → ops.concat_and_cache_mla(1118-1135) → cache_kernels.cu(414/462/516)
- backend 族：`vllm/v1/attention/backends/mla/`（~25 文件：flashmla、cutlass、flashinfer_mla
  (+sparse SM90/SM120/DSv4)、triton_mla、sparse indexer、sparse_swa）
- 选择：platforms/cuda.py MLA 链（SM100: FLASHINFER_MLA → TOKENSPEED_MLA → CUTLASS_MLA → …；
  SM120: TRITON_MLA first；fp8 KV 偏向 FlashInfer sparse）
- 测试：tests/kernels/attention/（flashmla、cutlass_mla、sparse parity）、
  tests/v1/attention/test_sparse_mla_kv_cache_layout.py

**驱动问题**
1. MLA cache 为什么是 [B, block, 1, kv_lora_rank+qk_rope_dim] 单张量？对比 MHA 的 K/V 分离，
   DeepSeek 级参数下压缩比多少？
2. concat_and_cache_mla concat 的是什么？为什么 MLA 不需要 reshape_and_cache_flash？
3. page_size_padded 对齐为什么存在？
4. SM100 上 fp8 KV 为什么偏向 FlashInfer sparse MLA？
5. sparse MLA 的 indexer 与 KV cache 的关系（索引器存哪）？

**实验**：若有 80G+ 卡：跑一个 DeepSeek-V2/V3 级 MLA 小模型，记录所选 backend 与 cache shape

---

## D21 上午：Spec Decode 专题 / 下午：复盘

**代码索引（上午）**
- `vllm/v1/spec_decode/eagle.py`（V1 EagleProposer）；`vllm/v1/worker/gpu/spec_decode/`
  （eagle/、mtp/、rejection_sampler.py、speculator.py、dflash/dspark）
- draft 元数据：backend.py build_for_drafting(735)、flash_attn.py(1037-1093)、
  update_draft_decode_metadata；drafter backend 初始化(7083-7093)、分离 builder 条件(6976-6998)
- 拒绝语义：gpu_model_runner.py 5127-5225（num_rejected_tokens_gpu）+ scheduler 2057-2066
  （num_computed_tokens 回滚）——KV 内容不回滚，槽位下步覆写
- MTP/EAGLE × hybrid：_update_states_after_model_execute(1564)、drop_eagle_block、
  eagle_group_ids、num_reprefillable_tokens（串联 W2）
- 测试：test_eagle_draft_attn_metadata.py、test_draft_attention_backend_override.py

**驱动问题**
1. draft 与 target 的 KV 在同一个 block pool 吗？is_eagle_group 标注怎么来的？
2. 被拒绝 token 的"脏 KV"为什么可以不清理？谁在哪个层面回滚 num_computed_tokens？
3. build_for_drafting 的 batch 形状与普通 decode 差异？
4. rejection sampler 在哪一层（worker）执行？与 metadata 的关系？
5. MTP 多模块的 num_reprefillable_tokens 怎么影响 cache_blocks？

**下午复盘**：`/kv-quiz week3` → 错题入薄弱点；**产出 W3 里程碑**：backend/kernel 对比报告写入 labs/；
`/kv-progress` 盘点。

**周验收**：里程碑 3 达成。
