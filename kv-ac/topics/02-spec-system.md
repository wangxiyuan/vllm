# 02 · KVCacheSpec 体系与混合分组（W1-D1）

## 核心问题
1. spec 为什么 per-layer？hybrid 模型怎么归组？
2. UniformTypeKVCacheSpecs 解决什么？EAGLE group 怎么标注？
3. 各 spec 的字段语义与 0 字节特例？

## 代码索引
- `vllm/v1/kv_cache_interface.py`：KVCacheSpec(156)/AttentionSpec(485)/FullAttentionSpec(549)/
  MLA(656)/RSWA(739)/ChunkedLocal(783)/SWA(823)/CircularBuffer(882)/SWA-MLA(922)/Mamba(1032)/
  EncoderOnly(1149)/Cross(1156)/Sink(1166)/UniformType(1223)/GroupSpec(1436)/Config(1457)
- `vllm/v1/kv_cache_spec_registry.py`、`vllm/v1/core/kv_cache_utils.py` get_kv_cache_groups(2264)

## 种子结论
- 归组启发式：同 spec 且同 manager 的组合并；page size 统一（unify_kv_cache_spec_page_size）；
  full attention 组排最前；三条路径 uniform / packed / GLM-5.3 特例；1.5x group-size 启发式(1525)。
- FullAttentionSpec 携带 sliding_window/chunk_size/non_causal 是为了 hybrid 管理器被禁用
  （--disable-hybrid-kv-cache-manager）时能退化成统一 FullAttention 处理。
- EncoderOnlyAttentionSpec 0 字节：层不存 AR KV，但仍进 config 以维持 layer→group 映射完整性。
- MambaSpec 直接继承 KVCacheSpec（非 AttentionSpec）：无 K/V 语义，shapes/dtypes 是
  (conv_state, ssm_state) 元组，bind 时切页成多视图。
- KVCacheSpecKind 供事件注记与可缓存性判断消费（take_events 里 per-group 填充）。

## 沉淀区
<!-- /kv-qa 增量 -->
