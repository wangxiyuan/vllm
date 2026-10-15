# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invariants of the golden KV cache grouping plans.

The plans in ``kv_plans/plans/`` come from real engine starts
(``kv_plans/check_plans.py``, on GPUs). These CPU tests read only the plan
files, so they fail only when a plan changes.
"""

import pytest

from tests.v1.core.kv_plans.cases import CASES, KVPlanCase
from tests.v1.core.kv_plans.plan_utils import (
    PLANS_DIR,
    layer_names,
    load_plan,
    render_plan,
)

MODELS = sorted({case.model for case in CASES})
PLANS = {name: entry for model in MODELS for name, entry in load_plan(model).items()}
PLANNED = [case for case in CASES if case.name in PLANS]

# Speculators known to regroup the target layers, with the reason.
DSV41_DSPARK = (
    "3 sliding-window draft layers resize the default heuristic's groups "
    "(vllm-project/vllm#58638)"
)
KNOWN_REGROUPING = {
    "dsv41_flash-tp1-dspark": DSV41_DSPARK,
    "dsv41_flash-tp4-dspark": DSV41_DSPARK,
}


def _grouping(case: KVPlanCase) -> list[list[str]]:
    return [layer_names(group) for group in PLANS[case.name]["structure"]]


@pytest.mark.parametrize("model", MODELS)
def test_plan_is_canonical(model: str):
    """Plans list only known cases, in the form ``check_plans.py`` writes."""
    path = PLANS_DIR / f"{model}.yaml"
    if not path.exists():
        pytest.skip(f"{model} has no plan")
    plan = load_plan(model)
    assert set(plan) <= {case.name for case in CASES}
    assert render_plan(model, plan) == path.read_text()


@pytest.mark.parametrize(
    "case",
    [case for case in PLANNED if case.speculator and case.base in PLANNED],
    ids=lambda case: case.name,
)
def test_speculator_keeps_target_grouping(case: KVPlanCase, request):
    """Draft layers may join groups or add one group of their own, but must not
    regroup the target layers."""
    if reason := KNOWN_REGROUPING.get(case.name):
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
    grouping = _grouping(case)
    base_grouping = _grouping(case.base)
    target_layers = {name for layers in base_grouping for name in layers}
    target_grouping = [[n for n in layers if n in target_layers] for layers in grouping]
    assert [layers for layers in target_grouping if layers] == base_grouping
    assert len(grouping) <= len(base_grouping) + 1


@pytest.mark.parametrize(
    "case",
    [case for case in PLANNED if case.tp > 1 and case.pp == 1 and case.dcp == 1],
    ids=lambda case: case.name,
)
def test_tensor_parallelism_keeps_grouping(case: KVPlanCase):
    tp1 = KVPlanCase(case.model, precision=case.precision, speculator=case.speculator)
    if tp1 not in PLANNED:
        pytest.skip(f"{tp1.name} has no plan")
    assert _grouping(case) == _grouping(tp1)
