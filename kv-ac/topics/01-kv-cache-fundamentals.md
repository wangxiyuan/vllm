# 01 · KV Cache 基础与分页思想（W1-D1）

## 核心问题
1. KV cache 存什么、为什么必须有？大小公式？
2. PagedAttention 的三个核心机制（分页/共享/CoW）分别解决什么浪费？
3. vLLM v1 与论文时代的实现差异有哪些（本仓库视角）？

## 代码索引
- `vllm/v1/kv_cache_interface.py`（spec 全家）
- `vllm/config/cache.py`（block_size、cache_dtype、enable_prefix_caching）

## 种子结论（读码后修正/补充）
- 每层每 token KV 字节 = 2(K/V) × kv_heads × head_size × dtype_bytes（MHA→GQA 少在 kv_heads；
  MLA 是 latent+rope 单张量，见 topics/20）。fp8 严格减半（无 scale 额外存储，scale 是 per-layer buffer）。
- block_size=16 权衡：太小→元数据/页表开销与 kernel launch 效率差；太大→内部碎片 + 细粒度
  命中率下降 + CoW 拷贝代价大。本仓库还存在 manager block ≠ kernel block 的二次切分。
- 论文时代的 v0 CUDA PagedAttention kernel 已从本仓库 csrc 移除（仅 CPU/XPU/ROCm 保留
  `ops/paged_attn.py` wrapper）；CUDA 主力是 vllm-flash-attn / FlashInfer / Triton。
- prefix caching 从"radix tree"演进为 hash 链 + BlockPool（`kv_cache_utils.py`）。

## 沉淀区（问答增量，最新在上）
<!-- /kv-qa 会往这里追加 -->
