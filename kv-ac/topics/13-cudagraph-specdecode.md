# 13 · CUDA Graph 不变量与 Spec Decode（W3-D18/D21）

## 核心问题
1. CG 四级支持与地址稳定不变量？
2. spec decode 的 KV 语义：为什么拒绝 token 不回滚 KV？
3. Mamba 为什么例外？

## 代码索引
- `vllm/v1/attention/backend.py` CGSupport(559)/build_for_drafting(735)
- `vllm/v1/worker/gpu/cudagraph_utils.py`(142)、`gpu/block_table.py`(182-238/326-328)
- `vllm/v1/spec_decode/eagle.py`、`vllm/v1/worker/gpu/spec_decode/`
- 回滚：`gpu_model_runner.py` 5127-5225 + `scheduler.py` 2057-2066、1564

## 种子结论
- AttentionCGSupport：ALWAYS / UNIFORM_BATCH（批内容同构即可）/ UNIFORM_SINGLE_TOKEN_DECODE
  （仅 decode 单 token 形状）/ NEVER（必须 piecewise capture）。connector layerwise 会要求
  piecewise（requires_piecewise_for_cudagraph，base.py:658）。
- 地址稳定：捕获期绑定的张量（page table、slot mapping、persistent batch 字段）回放期必须
  同地址，靠「持久张量 + 原地更新」实现；batch 缩小用 pad（PAD_SLOT_ID=-1，idx_mapping=-1）。
- 拒绝 token：KV 已写入但语义作废——不清理、不回滚块表，仅把 scheduler/worker 两侧
  num_computed_tokens 减掉 num_rejected_tokens，下一步这些槽位被覆写。命中统计因此要
  谨慎对待未验证尾部（cache_blocks 只缓存 verified tokens）。
- Mamba 例外：conv/ssm 状态是循环状态，覆写不可逆 → 显式 checkpoint（prefill 后）+ 回滚；
  _update_states_after_model_execute(1564) 提交/回退，num_speculative_blocks 预留草案块。
- draft 元数据：build_for_drafting 构建 draft 形状的 metadata；draft/target 头数不同时
  分离 builder（runner 6976-6998）。

## 沉淀区
