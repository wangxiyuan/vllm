# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache grouping owned by the Qwen4Exp model."""

from collections import defaultdict

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


class Qwen4ExpKVCacheGrouping:
    """Group the KV caches by the hybrid layer period.

    Each ``full_attention_interval`` layers hold ``interval - 1`` GDN layers and
    one QSA layer. Groups, in order of first registration:

    - GDN states: one group per position in the period.
    - PLE state: its own group, since it is TP-replicated.
    - QSA raw key rings: one scratch group.
    - QSA main and compressed caches: one group sharing a block table.

    MTP layers are QSA layers and join the QSA groups. Every group keeps the
    ``UniformTypeKVCacheSpecs`` representation of the default packed path.
    """

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_outermost:
            return get_kv_cache_groups(vllm_config, kv_cache_spec)

        period = vllm_config.model_config.hf_text_config.full_attention_interval
        gdn: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
        buckets: dict[str, dict[str, KVCacheSpec]] = {}
        for name, spec in kv_cache_spec.items():
            if isinstance(spec, MambaSpec):
                role = "ple" if spec.tp_replicated else "gdn"
            elif isinstance(spec, CircularBufferSpec):
                role = "raw"
            elif isinstance(spec, FullAttentionSpec) and not isinstance(
                spec, HiddenStateCacheSpec
            ):
                role = "qsa"
            else:
                return get_kv_cache_groups(vllm_config, kv_cache_spec)
            if role == "gdn":
                gdn[extract_layer_index(name) % period][name] = spec
            buckets.setdefault(role, {})[name] = spec

        groups = []
        for role, specs in buckets.items():
            parts = [gdn[pos] for pos in sorted(gdn)] if role == "gdn" else [specs]
            for part in parts:
                uniform_spec = UniformTypeKVCacheSpecs.from_specs(part)
                assert uniform_spec is not None, f"{role} layers must share a type"
                groups.append(KVCacheGroupSpec(list(part), uniform_spec))
        return groups
