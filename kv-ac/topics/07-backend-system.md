# 07 · Attention Backend 体系与选择（W1-D6）

## 核心问题
1. 选择的输入维度与优先级链？
2. 各 backend 的布局/能力约束？
3. metadata builder 每步构建什么？

## 代码索引
- `vllm/v1/attention/backend.py`（ABC 58/Builder 596/CGSupport 559/Impl 907）
- `vllm/v1/attention/selector.py`(105)、`backends/registry.py`(34)、`platforms/cuda.py`(80-178, 507-524)
- 代表实现：flash_attn/flashinfer/triton_attn/flex_attention/composite、mla/ 目录

## 种子结论
- 输入维度：GPU 代际（SM90/SM100/SM120）、是否 MLA、cache_dtype、hybrid 组成、
  layout、用户 flag（backend_per_kind、--block-size）、因果性。
- 默认（CUDA）：SM100 非 MLA → FLASHINFER（非因果偏 FLASH_ATTN）；SM90+ → FLASH_ATTN →
  FLASHINFER → TRITON → FLEX；MLA SM100 → FLASHINFER_MLA 系；SM120 → TRITON_MLA。
- 用户 --block-size 不兼容时 backend 会静默降级并 warn（cuda.py:507-524）。
- 约束速记：FlashInfer TRT-LLM decode 强制 HND(2503-2506)；Triton layout-agnostic；
  FlexAttention 仅 LBNHC；复合 backend（composite）服务多 group 不同 kind。
- Builder 每步产出 CommonAttentionMetadata（query 起点/长度、page table tensor、
  cu_seqlens 等），CUDA graph 下有 build_for_cudagraph_capture 与 update_block_table。

## 沉淀区
