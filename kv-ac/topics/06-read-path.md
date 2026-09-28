# 06 · 读取路径与 Slot Mapping（W1-D5）

## 核心问题
1. page table 与 slot_mapping 的分工？
2. slot 换算全公式（含 kernel block 切分）？
3. CUDA graph 下的 PAD 语义？

## 代码索引
- `vllm/v1/worker/gpu/block_table.py` BlockTables(17)/_gather_block_tables_kernel(242)/
  _compute_slot_mappings_kernel(282)/PAD(326-328)/地址稳定(182-238)
- `vllm/v1/worker/block_table.py`（V1 对应物）
- `vllm/v1/worker/gpu_model_runner.py` _get_slot_mappings(4065)
- `vllm/v1/attention/backend.py` CommonAttentionMetadata(386)

## 种子结论
- 写用 slot_mapping（每 token 的物理槽位 = block_id×kernel_block_size + offset），
  读用 block_table_tensor（每请求的块 id 列表，kernel 循环展开 K/V 页）。
- 换算链：position → kernel_block_idx = pos//kernel_bs → block_number = page_table[req][idx]
  → slot = block_number×kernel_bs + pos%kernel_bs。manager block 与 kernel block 不同时，
  page_table 的编号是 kernel block 粒度（blocks_per_kv_block 重映射 b*bpk+k）。
- CUDA graph：批形状固定，多余 token 位填 -1（PAD_SLOT_ID），kernel guard 跳过；
  持久张量地址必须跨步稳定（注释 182-238）。
- per-group slot mapping（不同 group 块大小不同）与 per-layer dict（forward context 用）并存。
- CP/PCP interleaving 会改变 token→position 的对应，slot kernel 内处理。

## 沉淀区
