# 04 · 物理分配与 KVCacheLayout（W1-D3）

## 核心问题
1. 为什么 flat int8 buffer + as_strided？
2. [L,B,H,N,C] 各维含义？布局排列怎么影响 kernel？
3. manager block 与 kernel block 切分的约束？

## 代码索引
- `vllm/v1/worker/utils.py` allocate_kv_cache(389)/prepare_kernel_block_sizes(458)/bind_kv_cache(591)
- `vllm/v1/kv_cache_interface.py` compute_layer_kv_cache_shape_bytes(295)/compute_layout_strides(314)/
  create_kv_cache_views(353)/报错建议(383-396)
- `vllm/v1/kv_cache_layout.py`

## 种子结论
- 单块大 int8 buffer 的理由：①一次 cudaMalloc，减少碎片与分配器压力；②as_strided 让不同
  layer/group 共享任意的逻辑形状而物理连续；③跨 rank/跨实例的块编号偏移可移植（offload 的
  canonical mapping 依赖）；④便于整块 zeroing 与镜像。
- 逻辑视图恒为 [B,H,N,C]（C 以字节；MHA 的 K/V 共享 C=2×head_size），物理顺序由
  KVCacheLayout 排列族决定：LBHNC≈HND（头在外），LBNHC≈NHD（token 在外）；FlexAttention
  只吃 LBNHC；FlashInfer TRT-LLM decode 要 HND。
- kernel block 切分（如 256→64）要求物理页 dense unpadded，否则 create_kv_cache_views 直接
  raise 并建议 VLLM_KV_CACHE_LAYOUT=LBNHC；Mamba 页不参与切分。
- bind_kv_cache 把视图挂到 layer 的 forward context；mamba 在 bind 时把页切成 conv/ssm 视图。

## 沉淀区
