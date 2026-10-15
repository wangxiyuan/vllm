# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden KV cache grouping plans: rendering, parsing and layer-name folding.

Each model has one plan file, ``plans/<model>.yaml``. It lists each distinct
grouping structure once and maps each case to it; byte sizes are left out, so
only grouping changes show. ``check_plans.py`` produces the structures from real
engine starts on GPUs; the CPU tests check invariants on the plan files alone.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from tests.v1.core.kv_plans.cases import CASES
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.core.kv_cache_planning import (
    _project_kv_cache_groups_to_worker,
    get_kv_cache_groups_from_builder,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

PLANS_DIR = Path(__file__).parent / "plans"


def global_kv_cache_groups(
    builder: KVCacheConfigBuilder,
    vllm_config: Any,
    stage_specs: list[dict[str, KVCacheSpec]],
) -> list[KVCacheGroupSpec]:
    """Group all stages' layers as ``get_kv_cache_configs`` does.

    ``check_plans.py`` checks the result against the groups the engine
    allocated on each worker.
    """
    merged: dict[str, KVCacheSpec] = {}
    for stage in stage_specs:
        merged.update(stage)
    extra_retained_tokens = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    for name, spec in merged.items():
        if isinstance(spec, SlidingWindowSpec):
            merged[name] = replace(spec, extra_retained_tokens=extra_retained_tokens)
    return get_kv_cache_groups_from_builder(builder, vllm_config, merged)


def describe_case(
    groups: list[KVCacheGroupSpec], stage_specs: list[dict[str, KVCacheSpec]]
) -> dict[str, Any]:
    """A case's plan entry: its grouping structure and, with PP, how many layers
    of each group each stage holds."""
    case: dict[str, Any] = {"structure": [_describe_group(g) for g in groups]}
    if len(stage_specs) > 1:
        case["pp_stage_layers"] = [
            [len(g.layer_names) for g in _project_kv_cache_groups_to_worker(groups, s)]
            for s in stage_specs
        ]
    return case


def render_plan(model: str, cases: dict[str, dict[str, Any]]) -> str:
    """Render a model's plan, listing each distinct structure once under the
    name of the first case that uses it."""
    entries: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    for case in CASES:
        if case.name not in cases:
            continue
        structure = cases[case.name]["structure"]
        short_name = case.name.removeprefix(f"{model}-")
        name = next(
            (n for n, known in structures.items() if known == structure), short_name
        )
        structures.setdefault(name, structure)
        entries[short_name] = {"structure": name, "groups": len(structure)}
        if "pp_stage_layers" in cases[case.name]:
            entries[short_name]["pp_stage_layers"] = cases[case.name]["pp_stage_layers"]
    return yaml.safe_dump(
        {"cases": entries, "structures": structures},
        sort_keys=False,
        width=1000,
        default_flow_style=None,
    )


def load_plan(model: str) -> dict[str, dict[str, Any]]:
    """Parse a model's plan into case entries keyed by full case name."""
    path = PLANS_DIR / f"{model}.yaml"
    if not path.exists():
        return {}
    plan = yaml.safe_load(path.read_text())
    cases = {}
    for short_name, entry in plan["cases"].items():
        case = {"structure": plan["structures"][entry["structure"]]}
        if "pp_stage_layers" in entry:
            case["pp_stage_layers"] = entry["pp_stage_layers"]
        cases[f"{model}-{short_name}"] = case
    return cases


def layer_names(group: dict[str, Any]) -> list[str]:
    """Expand a described group's folded layer names, in their original order."""
    ranked = []
    for rank, folded in enumerate(group["layers"]):
        head, brace, rest = folded.partition("{")
        if not brace:
            ranked.append((rank, folded))
            continue
        runs, _, tail = rest.partition("}")
        for run in runs.split(","):
            first, _, rest = run.partition("-")
            last, _, step = rest.partition(":")
            for i in range(int(first), int(last or first) + 1, int(step or 1)):
                ranked.append((rank, f"{head}{i}{tail}"))
    if group.get("layer_order") == "index_major":
        ranked.sort(key=lambda r: (_layer_index(r[1]), r[0]))
    return [name for _, name in ranked]


def _layer_index(name: str) -> int:
    return next((int(p) for p in name.split(".") if p.isdigit()), -1)


def _describe_group(group: KVCacheGroupSpec) -> dict[str, Any]:
    spec = group.kv_cache_spec
    specs = (
        list(spec.kv_cache_specs.values())
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else [spec]
    )
    entry: dict[str, Any] = {
        "kind": " + ".join(dict.fromkeys(type(s).__name__ for s in specs))
    }
    windows = sorted(
        {w for s in specs if (w := getattr(s, "sliding_window", None)) is not None}
    )
    if windows:
        entry["sliding_window"] = windows[0] if len(windows) == 1 else windows
    if group.is_eagle_group:
        entry["eagle"] = True
    if group.role is not KVCacheGroupRole.DEFAULT:
        entry["role"] = group.role.value
    if group.host_resident:
        entry["host_resident"] = True
    if not group.enable_kv_transfer:
        entry["kv_transfer"] = False
    entry.update(_fold_layer_names(group.layer_names))
    return entry


def _fold_layer_names(names: list[str]) -> dict[str, Any]:
    """Fold layer names by their first numeric component, keeping their order:
    ``model.layers.{0-8:4}.attn`` is layers 0, 4 and 8. Names that interleave
    several templates by layer index add ``layer_order: index_major``; names
    that fold neither way stay as they are."""
    templates: dict[str, list[int]] = {}
    keys = []
    for name in names:
        parts = name.split(".")
        pos = next((i for i, part in enumerate(parts) if part.isdigit()), None)
        if pos is None:
            template, index = name, -1
        else:
            template = ".".join([*parts[:pos], "{}", *parts[pos + 1 :]])
            index = int(parts[pos])
        templates.setdefault(template, []).append(index)
        keys.append((template, index))

    rank = {template: i for i, template in enumerate(templates)}
    template_major = [(t, i) for t, indices in templates.items() for i in indices]
    index_major = sorted(keys, key=lambda key: (key[1], rank[key[0]]))
    if keys == template_major:
        entry: dict[str, Any] = {}
    elif keys == index_major:
        entry = {"layer_order": "index_major"}
    else:
        return {"layers": list(names)}
    entry["layers"] = [
        template
        if indices == [-1]
        else template.format(
            indices[0] if len(indices) == 1 else "{" + _fold_indices(indices) + "}"
        )
        for template, indices in templates.items()
    ]
    return entry


def _fold_indices(indices: list[int]) -> str:
    """Fold indices into ``a``, ``a-b`` (step 1) and ``a-b:s`` runs."""
    runs = []
    start = 0
    while start < len(indices):
        end = start + 1
        step = indices[end] - indices[start] if end < len(indices) else 0
        while end < len(indices) and indices[end] - indices[end - 1] == step > 0:
            end += 1
        first, last = indices[start], indices[end - 1]
        if end - start >= 2 and step == 1:
            runs.append(f"{first}-{last}")
        elif end - start >= 3:
            runs.append(f"{first}-{last}:{step}")
        else:
            runs.append(str(first))
            end = start + 1
        start = end
    return ",".join(runs)
