# 12 · 量化 KV Cache（W3-D17）

## 核心问题
1. cache_dtype 的值域与对应 kernel？
2. k/v scale 从哪来、两条流向？
3. per-tensor/per-head/per-token-head dispatch？nvfp4？

## 代码索引
- `vllm/v1/kv_cache_interface.py` KVQuantMode(39)
- `vllm/config/cache.py` 39-57
- `vllm/model_executor/layers/attention/attention.py` 136-153/157/221-234
- `csrc/libtorch_stable/cache_kernels.cu`（写侧量化分支）、nvfp4_kv_cache_kernels.cu

## 种子结论
- 值域：fp8(=e4m3)/fp8_e4m3/fp8_e5m2/nvfp4/nvfp4_4over6/nvfp4_ds_mla/int8/int4/fp8_per_token_head。
- scale 来源：checkpoint 带 BaseKVCacheMethod（如 compressed-tensors fp8）→ 从权重加载；
  否则 1.0 buffer。e5m2 动态范围大通常免 scale；e4m3 必须配 scale。
- 两条流：写侧 kernel 内量化（reshape_and_cache_flash 的 k_scale/v_scale 参数）；
  读侧 k_descale/v_descale 传入 backend attention kernel 反量化。
- 粒度 dispatch：kv_scale_stride==0 → per-tensor fast path；stride>0 → per-head（[num_heads]）；
  per-token-head 是独立 KVQuantMode，走专门 kernel。
- 冲突检查：fp8_e5m2 与 fp8 checkpoint 共存需 quant 方法声明无 kv_cache_scheme(221-234)。
- 支持矩阵：supports_kv_cache_dtype 按 backend 声明；nvfp4 主要 SM100+ FlashInfer 系。

## 沉淀区
