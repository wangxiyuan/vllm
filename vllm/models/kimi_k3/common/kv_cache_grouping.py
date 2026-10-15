# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache grouping owned by the Kimi-K3 model."""

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


class KimiK3KVCacheGrouping:
    """Group the target layers as if there were no drafter.

    The default grouping sizes every group by the smallest set of compatible
    layers. A DSpark drafter adds 5 sliding-window layers, which cuts the
    23-layer KDA and MLA sets into 5-layer groups: 20 groups instead of 4
    (vllm-project/vllm#58638). Draft layers, numbered after the target's, are
    grouped on their own instead.
    """

    @classmethod
    def get_kv_cache_groups(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[KVCacheGroupSpec]:
        from vllm.v1.core.kv_cache_planning import (
            get_kv_cache_groups,
            unify_kv_cache_spec_page_size,
        )

        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not layout.is_block_outermost:
            # All groups share one page size here, so unify target and draft
            # layers together before grouping them apart.
            kv_cache_spec = unify_kv_cache_spec_page_size(kv_cache_spec)
        num_target_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
        target = {
            name: spec
            for name, spec in kv_cache_spec.items()
            if extract_layer_index(name) < num_target_layers
        }
        draft = {
            name: spec for name, spec in kv_cache_spec.items() if name not in target
        }
        groups = get_kv_cache_groups(vllm_config, target)
        if draft:
            groups += get_kv_cache_groups(vllm_config, draft)
        return groups
