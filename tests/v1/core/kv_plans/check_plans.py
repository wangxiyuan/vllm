# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the golden KV cache grouping plans against real engine starts.

Each case starts an engine from the model's config.json with dummy weights; the
original layer count is kept and the MoE expert count is cut to fit. The KV
cache groups come from the platform's builder, so a model's
``SupportsKVCacheGrouping`` hook is used. The script fails if a case's grouping
differs from ``plans/<model>.yaml``; ``--update`` rewrites the plan instead.

Run it on a node with 4 GPUs, from the repository root::

    python -m tests.v1.core.kv_plans.check_plans [--model M] [--case C] [--update]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path

from tests.v1.core.kv_plans.cases import CASES, KVPlanCase
from tests.v1.core.kv_plans.plan_utils import load_plan, render_plan

CASE_TIMEOUT_S = 30 * 60


def small_moe_hf_overrides(hf_config, min_experts: int):
    """Keep the layer count and cut the MoE experts, but keep at least
    ``min_experts``, since some models split experts over TP ranks."""
    from tests.models.utils import dummy_hf_overrides

    hf_config = dummy_hf_overrides(hf_config, use_original_num_layers=True)
    text_config = hf_config.get_text_config()
    for key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if getattr(text_config, key, min_experts) < min_experts:
            setattr(text_config, key, min_experts)
    return hf_config


def run_case(case: KVPlanCase, out: Path) -> None:
    """Start the engine for one case and write its plan entry to ``out``."""
    # The engine core runs in this process, so its planning call can be wrapped.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm.v1.engine.core as engine_core
    from tests.v1.core.kv_plans.plan_utils import (
        describe_case,
        global_kv_cache_groups,
    )
    from vllm import LLM
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
    from vllm.v1.core.kv_cache_planning import _project_kv_cache_groups_to_worker

    planned = []
    get_kv_cache_configs = engine_core.get_kv_cache_configs

    def recording_get_kv_cache_configs(vllm_config, kv_cache_specs, memory):
        configs = get_kv_cache_configs(vllm_config, kv_cache_specs, memory)
        planned.append((vllm_config, kv_cache_specs, configs))
        return configs

    engine_core.get_kv_cache_configs = recording_get_kv_cache_configs
    LLM(
        **case.llm_kwargs(),
        load_format="dummy",
        hf_overrides=partial(small_moe_hf_overrides, min_experts=case.tp),
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.7,
    )
    ((vllm_config, worker_specs, worker_configs),) = planned

    # TP and DCP ranks of one PP stage hold the same specs.
    stage_specs: list[dict] = []
    for specs in worker_specs:
        if specs not in stage_specs:
            stage_specs.append(specs)
    groups = global_kv_cache_groups(
        get_kv_cache_config_builder(vllm_config), vllm_config, stage_specs
    )
    for specs, config in zip(worker_specs, worker_configs):
        projected = _project_kv_cache_groups_to_worker(groups, specs)
        expected = [(g.layer_names, g.is_eagle_group) for g in projected]
        actual = [(g.layer_names, g.is_eagle_group) for g in config.kv_cache_groups]
        assert actual == expected, f"{case.name}: engine groups differ from the plan"
    out.write_text(json.dumps(describe_case(groups, stage_specs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Check only the cases of this model.")
    parser.add_argument("--case", help="Check only this case.")
    parser.add_argument("--update", action="store_true", help="Rewrite the plans.")
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = [
        case
        for case in CASES
        if (args.model is None or case.model == args.model)
        and (args.case is None or case.name == args.case)
    ]
    if args.out:
        (case,) = cases
        run_case(case, args.out)
        return

    # One process per case, so each engine starts on clean GPUs.
    failed, changed = [], []
    for model in dict.fromkeys(case.model for case in cases):
        plan = load_plan(model)
        for case in (c for c in cases if c.model == model):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "case.json"
                cmd = [sys.executable, "-m", __spec__.name, "--case", case.name]
                try:
                    subprocess.run(
                        [*cmd, "--out", str(out)], timeout=CASE_TIMEOUT_S, check=True
                    )
                    result = json.loads(out.read_text())
                except (subprocess.SubprocessError, OSError):
                    print(f"[FAILED] {case.name}", flush=True)
                    failed.append(case.name)
                    continue
            if plan.get(case.name) == result:
                print(f"[ok] {case.name}", flush=True)
                continue
            print(f"[CHANGED] {case.name}", flush=True)
            print(render_plan(model, {case.name: result}), flush=True)
            changed.append(case.name)
            plan[case.name] = result
        if args.update:
            (Path(__file__).parent / "plans" / f"{model}.yaml").write_text(
                render_plan(model, plan)
            )
    if failed or (changed and not args.update):
        sys.exit(f"Failed: {failed}; changed: {changed}")


if __name__ == "__main__":
    main()
