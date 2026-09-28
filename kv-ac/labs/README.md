# 实验手册

> 全程遵守仓库 AGENTS.md：**不要用系统 python3 / 裸 pip**，一切走 `uv` 与 `.venv/bin/python`。

## 0. 环境准备（D1 一次性完成，约 1h）

```bash
cd /home/wxy/code/vllm
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/lint.txt && pre-commit install
uv pip install -r requirements/test/cuda.in
```

冒烟验证：

```bash
.venv/bin/python -m pytest tests/v1/core/test_kv_cache_utils.py -x -q   # 单测通
.venv/bin/python -c "from vllm import LLM; print('import ok')"           # 可导入
```

## 1. 常用命令

| 目的 | 命令 |
|---|---|
| 核心 core 单测 | `.venv/bin/python -m pytest tests/v1/core/ -x -q` |
| prefix caching 单测 | `.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py -q` |
| worker/block table 单测 | `.venv/bin/python -m pytest tests/v1/worker/ -q` |
| backend parity | `.venv/bin/python -m pytest tests/v1/attention/test_attention_backends.py -q` |
| kernel 单测 | `.venv/bin/python -m pytest tests/kernels/test_cache_kernels.py -q` |
| connector 生命周期 | `.venv/bin/python -m pytest tests/v1/kv_connector/unit/test_kv_connector_lifecycle.py -q` |
| 写入 kernel 微基准 | `.venv/bin/python benchmarks/kernels/benchmark_reshape_and_cache_flash.py` |
| TP=2 起服务 | `vllm serve <model> --tensor-parallel-size 2` |
| MRV2 runner | 加 `VLLM_USE_V2_MODEL_RUNNER=1` |
| KV 布局实验 | 加 `VLLM_KV_CACHE_LAYOUT=LBNHC` |
| 量化 KV | 加 `--kv-cache-dtype fp8`（注意本仓库字段已改名 `cache_dtype`） |

## 2. PD 分离双实例模板（W4 用）

```bash
# 预填充实例 P
vllm serve <model> --port 8100 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda"}'
# 解码实例 D
vllm serve <model> --port 8200 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cuda"}'
# 前面挂 proxy（examples/disaggregated/disagg_proxy_demo.py）
```

## 3. 实验记录模板

```
## <日期> · <实验名>（对应 weekN-DX）
环境：GPU × N / 模型 / 关键 flags
步骤：…
观察（数字/日志摘录）：…
结论 & 疑问：…（疑问→/kv-qa）
```

## 4. 实验索引（按周）

- W1：D1 模型 spec 打印；D2 num_blocks 验算；D3 布局/stride 打印；D6 backend 选择观察
- W2：hit-rate 压测；OOM 触发 preemption trace；破坏性实验（改 free 顺序）
- W3：backend parity；cache_kernel 微基准；fp8 开关对比；TP=2 backend 对比
- W4：PD 双实例；CPU offload；mini connector；PR 实战

（实验记录追加在本文件下方）
