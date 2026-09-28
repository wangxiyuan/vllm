# 05 · 写入路径：token → 物理 KV（W1-D4）

## 核心问题
1. Attention.forward 到 CUDA 写内存的完整链路？
2. custom op（unified_kv_cache_update）为什么存在？
3. 写侧量化与 scale 分支？

## 代码索引
- `vllm/model_executor/layers/attention/attention.py` forward(488)/分发(724/767)/上下文解析(673-697)
- `vllm/v1/attention/backend.py` do_kv_cache_update 链(907)/MLA(1118-1135)
- `vllm/v1/attention/backends/flash_attn.py` do_kv_cache_update(1507)→ops(1532)
- `csrc/libtorch_stable/cache_kernels.cu` reshape_and_cache_flash_kernel(326)

## 种子结论
- 链路：forward(q,k,v) → unified_kv_cache_update（custom op，torch.compile 图内稳定）→
  从 forward_context 取 (attn_metadata, layer, kv_cache, layer_slot_mapping) →
  impl.do_kv_cache_update → FA: transpose(B,H,N,2D)→(B,N,H,2D) split K/V →
  ops.reshape_and_cache_flash(k,v,kc,vc,slot_mapping,dtype,k_scale,v_scale) →
  CUDA: slot=slot_mapping[t]（-1 跳过）、block=slot//bs、off=slot%bs → 向量化写。
- MLA 走 concat_and_cache_mla（latent+rope 拼接单张量）。
- 写侧 fp8 量化在 kernel 内完成；per-tensor scale（kv_scale_stride==0）走 fast path，
  per-head 走一般 path；nvfp4 有独立 kernel 族。
- _k_scale/_v_scale 是 per-layer buffer（attention.py:136-153），checkpoint 有
  BaseKVCacheMethod 时从权重加载，否则 1.0。

## 沉淀区
