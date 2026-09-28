# 03 · 显存 Profiling 与容量规划（W1-D2）

## 核心问题
1. available KV 显存怎么从 profile_run 推出？扣了哪些项？
2. num_blocks / bytes_per_block 怎么算？为何跨 rank 取 min？
3. 分组路径选择与 max_model_len auto-fit？

## 代码索引
- `vllm/v1/worker/gpu_worker.py` determine_available_memory(571)、653-660
- `vllm/v1/worker/gpu_model_runner.py` profile_run(6406)
- `vllm/v1/core/kv_cache_utils.py` get_kv_cache_configs(2628)/get_kv_cache_groups(2264)/
  get_kv_cache_config_from_groups(1633)/_get_kv_cache_bytes_per_block(1570)/auto-fit(2522)

## 种子结论
- 流程：load model → profile_run（max_num_tokens dummy 前向，含 activation 峰值）→
  释放 run 中间量 → 剩余显存 − 非 KV 常驻 − cudagraph 估算 → available_kv_cache_memory。
- num_blocks = available // bytes_per_block（跨 TP/PP rank 取 min，保持块编号全局一致）。
- bytes_per_block 由 spec 组的 page size 与 dtype 推（MLA/mamba 有专门分支 1570）。
- max_model_len 若放不下会 auto-fit 下调（2522），日志可见。
- 覆盖口子：--num-gpu-blocks-override；监控口子：SchedulerStats.kv_cache_usage。

## 沉淀区
