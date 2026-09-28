# Week 1（D1-D7）基础与全链路

> 周目标：画出「一个 token 从 scheduler 到物理 KV cache 写入、再被读出」的完整图；
> 手算 num_blocks 误差为 0；TP=2 跑通。对应讲义：topics/01~07。

---

## D1（9-28 周一）上午：论文基础 / 下午：spec 体系

**目标**：建立 KV cache 理论底座；默写 KVCacheSpec 继承树。

**材料**
- 论文《Efficient Memory Management for LLM Serving with PagedAttention》(SOSP'23)：
  精读 §3-4（分页、共享、CoW 动机），跳过与旧版实现绑定的细节
- 论文《FlashAttention-2》：只读「tiling 与 KV 的角色」部分
- 《Orca》(OSDI'22)：略读，知道 continuous batching 与 iteration-level scheduling 即可

**代码索引**（先通读，不必逐行）
- `vllm/v1/kv_cache_interface.py`：KVCacheSpec(156)、KVCacheSpecKind(137)、KVQuantMode(39)、
  AttentionSpec(485)、FullAttentionSpec(549)、MLAAttentionSpec(656)、RSWASpec(739)、
  ChunkedLocalAttentionSpec(783)、SlidingWindowSpec(823)、CircularBufferSpec(882)、
  SlidingWindowMLASpec(922)、MambaSpec(1032)、EncoderOnlyAttentionSpec(1149)、
  CrossAttentionSpec(1156)、SinkFullAttentionSpec(1166)、UniformTypeKVCacheSpecs(1223)、
  KVCacheGroupSpec(1436)、KVCacheConfig(1457)
- `vllm/v1/kv_cache_spec_registry.py`（spec→manager 注册机制）
- `vllm/config/cache.py`：CacheConfig 全文（block_size 73、prefix_match_unit 91、cache_dtype 123、
  enable_prefix_caching 142、prefix_caching_hash_algo 144、mamba_* 178-199、num_gpu_blocks 220）

**驱动问题**（先自答再读码）
1. 每层每 token 的 KV 字节数公式？GQA 相比 MHA 少在哪一项？fp8 是严格一半吗？
2. 默认 block_size=16 的权衡：太大/太小各损失什么？
3. 为什么 spec 是 per-layer 的？一个模型里多种 spec 并存（hybrid）的根源是什么？
4. FullAttentionSpec 为什么也有 sliding_window / attention_chunk_size 字段（hybrid 管理器被禁用时的 fallback 语义）？
5. MambaSpec 为什么直接继承 KVCacheSpec？它的 shapes/dtypes 元组对应什么状态？
6. UniformTypeKVCacheSpecs 与 KVCacheGroupSpec 的关系？"uniform type" 解决什么分组问题？
7. KVCacheSpecKind 被谁消费？（提示：block 事件与可缓存性）
8. EncoderOnlyAttentionSpec 为什么 0 字节还要进 KVCacheConfig？

**实验**（labs/README §0 环境准备后）
- 打印小模型（如 Qwen2.5-0.5B）的 `get_kv_cache_spec()` 输出，对照 spec 树
- 手算该模型每 token KV 字节数

**验收**：spec 层次图默写（画在 topics/02）；字节手算正确。

---

## D2 显存规划链路

**目标**：解释启动日志里每个 KV cache 数字；手算 num_blocks 误差 0。

**代码索引**
- `vllm/v1/worker/gpu_worker.py`：determine_available_memory(571)、653-660（KV 显存扣减）
- `vllm/v1/worker/gpu_model_runner.py`：profile_run(6406)、_dummy_run（is_profile=True）
- `vllm/v1/core/kv_cache_utils.py`：get_kv_cache_configs(2628)、get_kv_cache_groups(2264，含
  uniform / packed / GLM-5.3 三条路径)、get_kv_cache_config_from_groups(1633)、
  _get_kv_cache_bytes_per_block(1570)、unify_kv_cache_spec_page_size、
  _annotate_eagle_groups(2134)、max_model_len auto-fit(2522)

**驱动问题**
1. 为什么用 profile_run（真跑 dummy batch）而不是静态公式估显存？
2. available_kv_cache_memory 从 free-after-profile 里扣掉了什么？cudagraph 内存在哪一步估？
3. num_blocks 为什么跨 rank 取 min？不一致会怎样？
4. bytes_per_block 对 MLA / Mamba 怎么算？（_get_kv_cache_bytes_per_block）
5. 分组的三条路径（uniform/uniform-type-packed/特例）各自何时触发？1.5x group-size 启发式在哪？
6. EAGLE group 的 `_annotate_eagle_groups` 为什么被注释称为 "hacky"？位置法 fallback 是什么？

**实验**
- TP=2 起服务，读启动日志：per-layer KVCacheTensor、bytes、num_blocks；手算对照
- 改 `--gpu-memory-utilization` 观察线性关系；改 `--num-gpu-blocks-override` 覆盖行为

**验收**：验算误差为 0；能解释日志每个数字。

---

## D3 物理分配与布局

**目标**：画出「一个 block 的物理内存布局」图。

**代码索引**
- `vllm/v1/worker/utils.py`：allocate_kv_cache(389)（单 flat int8 buffer + as_strided）、
  prepare_kernel_block_sizes(458)、bind_kv_cache(591)/bind_kv_cache_to_layers(644)、request_memory(525)
- `vllm/v1/kv_cache_interface.py`：compute_layer_kv_cache_shape_bytes(295)、
  compute_layout_strides(314)、create_kv_cache_views(353)、383-396（dense unpadded 报错与
  `VLLM_KV_CACHE_LAYOUT=LBNHC` 建议）
- `vllm/v1/kv_cache_layout.py` 全文（[L,B,H,N,C] 排列族）
- `vllm/v1/worker/gpu_model_runner.py`：initialize_kv_cache(7332)、initialize_kv_cache_tensors(7260)、
  get_kv_cache_spec(7421)、kv_sharing 跳过与别名(7421-7447/7263-7267)

**驱动问题**
1. 为什么一个 flat int8 大 buffer + per-layer 视图，而不是每层独立 tensor？（≥3 个理由）
2. [L,B,H,N,C] 五维各是什么？LBHNC 与 LBNHC 分别对应哪种经典物理布局（HND/NHD）？
3. manager block 256 → kernel block 64 的切分为什么要 dense unpadded pages？谁会因此报错？
4. Mamba 页为什么不做 kernel block 切分？
5. C 维以字节为单位分配再 view 回 dtype——这个技巧解决什么？
6. kv_sharing（YOCO）的层为什么跳过分配、做别名？

**实验**
- python 打印每层 KVCacheTensor 的 size/shape/stride；切 VLLM_KV_CACHE_LAYOUT 对比

**验收**：画出 block 物理布局图（K/V 切分、MLA latent 两种）。

---

## D4 写入路径：token → 物理 cache

**目标**：从 Python 一路讲到 CUDA 写内存那一行。

**代码索引**
- `vllm/model_executor/layers/attention/attention.py`：__init__ 注册(444-448)、
  _init_kv_cache_quant(157)、forward(488)、get_kv_cache_spec(606)、
  unified_kv_cache_update 分发(724)、unified_attention_with_output(767)、fp8 冲突检查(221-234)
- `vllm/v1/attention/backend.py`：AttentionImpl.do_kv_cache_update 链(907 起)、
  MLA 基类 do_kv_cache_update → ops.concat_and_cache_mla(1118-1135)
- `vllm/v1/attention/backends/flash_attn.py`：do_kv_cache_update(1507)、transpose+split(1509 附近)、
  ops.reshape_and_cache_flash 调用(1532)
- `csrc/libtorch_stable/cache_kernels.cu`：reshape_and_cache_flash_kernel(326)、
  reshape_and_cache_kernel(266)、concat_and_cache_mla_kernel(414)/grouped(462)/DS(516)、
  C++ launcher(820/917)、nvfp4 dispatch(848+)

**驱动问题**
1. `unified_kv_cache_update` 这个 custom op 为什么存在？（torch.compile 拆图与上下文解析 673-697）
2. layer_slot_mapping 从哪个上下文对象来？
3. kernel 里 `slot_mapping[token]` 为 -1（PAD）时发生什么？
4. 写侧 fp8 量化在哪几行？per-tensor fast path 与 per-head path 的分支条件（kv_scale_stride）？
5. MLA 层写入与 MHA 写入的差异（concat 什么、cache 形状）？

**实验**：跑 `.venv/bin/python -m pytest tests/kernels/test_cache_kernels.py -q`

**验收**：口头复述全链路 ≥10 步，无断点。

---

## D5 读取路径与 slot mapping

**目标**：给定 (position, block_table, block_size) 手算 slot 无误。

**代码索引**
- `vllm/v1/worker/block_table.py`：BlockTable(57)、MultiGroupBlockTable(289)、
  ComputeSlotMappingKernel(398)、SlotMappingMode(52)
- `vllm/v1/worker/gpu/block_table.py`：BlockTables(17)、_gather_block_tables_kernel(242)、
  _compute_slot_mappings_kernel(282)、PAD 语义(326-328)、地址稳定注释(182-238)
- `vllm/v1/worker/gpu_model_runner.py`：_get_slot_mappings(4065)、per-layer dict 传递
- `vllm/v1/attention/backend.py`：CommonAttentionMetadata(386)（block_table_tensor 消费方）

**驱动问题**
1. page table 与 slot_mapping 谁服务「写」谁服务「读」？
2. kernel_block ≠ manager block 时的换算：pos//kernel_block → block_number → slot = block_number*kernel_block_size + offset，逐变量解释
3. CP（context parallel）interleaving 怎么影响 slot？
4. CUDA graph 下为什么 pad -1 而不是缩短 tensor？
5. per-group slot mapping 与 per-layer slot mapping 为什么都要有？

**实验**：跑 tests/v1/worker/test_gpu_block_table.py；写 10 行脚本手动推一个请求的 slot 序列

**验收**：手算 5 组 slot 全对。

---

## D6 backend 体系与选择

**目标**：对本机硬件说出默认 backend + 三个 fallback 理由。

**代码索引**
- `vllm/v1/attention/backend.py`：Backend ABC(58)、capability predicates、
  AttentionMetadataBuilder(596)、build_for_drafting(735)、AttentionCGSupport(559)、AttentionImpl(907)
- `vllm/v1/attention/selector.py`：get_attn_backend(105)、AttentionSelectorConfig(21)、
  backend_per_kind、get_mamba_attn_backend(228)
- `vllm/v1/attention/backends/registry.py`：AttentionBackendEnum(34)、MambaAttentionBackendEnum(203)
- `vllm/platforms/cuda.py`：优先级表(80-178)、block-size 静默降级警告(507-524)
- 代表 backend：flash_attn.py(286/601/815)、flashinfer.py(HND 强制 2503-2506、permute 2160)、
  triton_attn.py(layout-agnostic 395-421)、flex_attention.py(LBNHC only 130-133)、composite.py

**驱动问题**
1. backend 挑选的输入维度有哪些？（硬件代际/dtype/hybrid/layout/用户 flag/是否非因果）
2. SM100 非 MLA 默认 flashinfer、SM90 默认 flash_attn——各自底层 kernel 优势是什么？
3. `--block-size` 为什么会让 backend 静默降级？警告在哪？
4. composite backend 什么时候出现（多 group 不同 kind）？
5. AttentionMetadataBuilder 每步构建什么？quadratic/linear 元数据差异？

**实验**：跑 tests/v1/attention/test_attention_backends.py（parity）；TP=2 记录所选 backend 并解释

**验收**：本机 backend 决策树讲清。

---

## D7（10-4 周日）复盘日

- 上午：手画全链路大图（scheduler→worker→kernel→读回），对照 topics/00-code-map.md 的数据流图查漏
- 下午：`/kv-quiz week1`（15 题）→ 判卷 → 错题入薄弱点
- 收尾：`/kv-progress`；更新 progress.md 勾选；下周预习 W2 文件清单

**周验收**：里程碑 1 达成（见 progress.md）。
