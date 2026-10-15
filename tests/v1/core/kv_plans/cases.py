# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine configurations covered by the golden KV cache grouping plans."""

import copy
from dataclasses import dataclass, replace
from typing import Any

MODELS: dict[str, dict[str, Any]] = {
    "dsv41_flash": {"model": "deepseek-ai/DeepSeek-V4.1-Flash"},
    "qwen38_flash_next": {"model": "Qwen/Qwen3.8-Flash-Next"},
    "kimi_k3": {"model": "moonshotai/Kimi-K3", "trust_remote_code": True},
}

PRECISIONS: dict[str, dict[str, Any]] = {
    "indexer_mxfp4": {"attention_config": {"indexer_kv_dtype": "mxfp4"}},
}

SPECULATORS: dict[str, dict[str, Any]] = {
    "dsv41_flash-dspark": {
        "method": "dspark",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "num_speculative_tokens": 7,
    },
    "qwen38_flash_next-mtp": {"method": "mtp", "num_speculative_tokens": 1},
    "kimi_k3-dspark": {
        "method": "dspark",
        "model": "RedHatAI/Kimi-K3-speculator.dspark",
        "num_speculative_tokens": 7,
    },
}


@dataclass(frozen=True)
class KVPlanCase:
    model: str
    tp: int = 1
    pp: int = 1
    dcp: int = 1
    precision: str | None = None
    speculator: str | None = None

    @property
    def name(self) -> str:
        parts = [self.model, f"tp{self.tp}"]
        if self.pp > 1:
            parts.append(f"pp{self.pp}")
        if self.dcp > 1:
            parts.append(f"dcp{self.dcp}")
        parts += [p for p in (self.precision, self.speculator) if p]
        return "-".join(parts)

    @property
    def base(self) -> "KVPlanCase":
        """The same case without the speculator."""
        return replace(self, speculator=None)

    def llm_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            **MODELS[self.model],
            "tensor_parallel_size": self.tp,
            "pipeline_parallel_size": self.pp,
            "decode_context_parallel_size": self.dcp,
        }
        if self.precision:
            kwargs |= PRECISIONS[self.precision]
        if self.speculator:
            kwargs["speculative_config"] = SPECULATORS[
                f"{self.model}-{self.speculator}"
            ]
        return copy.deepcopy(kwargs)


CASES = [
    KVPlanCase("dsv41_flash"),
    KVPlanCase("dsv41_flash", tp=4),
    # PP: a non-first stage fails `residual is not None` in the model forward.
    # DCP: the two-level indexer candidate filtering does not support DCP.
    KVPlanCase("dsv41_flash", precision="indexer_mxfp4"),
    KVPlanCase("dsv41_flash", speculator="dspark"),
    KVPlanCase("dsv41_flash", tp=4, speculator="dspark"),
    KVPlanCase("qwen38_flash_next"),
    KVPlanCase("qwen38_flash_next", tp=4),
    # Qwen4Exp's N-gram PLE embedding does not support PP.
    KVPlanCase("qwen38_flash_next", speculator="mtp"),
    KVPlanCase("qwen38_flash_next", tp=4, speculator="mtp"),
    KVPlanCase("kimi_k3"),
    KVPlanCase("kimi_k3", tp=4),
    KVPlanCase("kimi_k3", tp=2, pp=2),
    KVPlanCase("kimi_k3", speculator="dspark"),
    KVPlanCase("kimi_k3", tp=4, speculator="dspark"),
]
