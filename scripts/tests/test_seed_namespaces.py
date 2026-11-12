"""Historical namespaces are inputs, not tunable performance parameters."""
import numpy as np
from scripts import simulation


def test_unified_d5_retains_archived_seed_streams():
    unified=simulation.load_config(simulation.DEFAULT_CONFIG)
    legacy=dict(unified,namespace="R61-D5-BALANCED-v1",namespace_by_dgp={})
    for case in simulation.expand_cases(unified,{"main","sensitivity","kmax"}):
        ns=simulation.case_namespace(unified,case)
        if case["dgp"]=="D5":
            assert ns==simulation.case_namespace(legacy,case)
        else:
            assert ns=="R61-FORMAL-v1"
        label=f"{case['stage']}|{case['cell']}|R000001"
        for suffix in ("DATA|"+label,"CHAIN1|"+label,"CHAIN2|"+label,"CHAIN3|"+label):
            assert simulation.deterministic_seed(ns,suffix)==simulation.deterministic_seed(
                "R61-D5-BALANCED-v1" if case["dgp"]=="D5" else "R61-FORMAL-v1",suffix)


def test_legacy_config_without_map_keeps_namespace():
    assert simulation.case_namespace({"namespace":"historical"},{"dgp":"D5"})=="historical"


def test_weighted_prefix_cost_is_translation_invariant():
    values=np.array([0.,1.,-1.,2.,-2.,0.,1.,-1.]);weights=np.ones(8)
    cost=simulation.WeightedL2(2).fit(np.column_stack((1e8+values,weights)))
    assert abs(cost.error(0,8)-float(np.sum((values-values.mean())**2)))<1e-12
    assert simulation.pelt_bic(values,weights,2)==simulation.pelt_bic(1e8+values,weights,2)


def test_explicit_pilot_namespace_overrides_every_dgp_without_mutating_defaults():
    cfg=simulation.load_config(simulation.DEFAULT_CONFIG)
    pilot=simulation.override_seed_namespace(cfg,"explicit-pilot")
    for case in simulation.expand_cases(pilot,{"main","sensitivity","kmax"}):
        assert simulation.case_namespace(pilot,case)=="explicit-pilot"
    assert simulation.case_namespace(cfg,{"dgp":"D5"})=="R61-D5-BALANCED-v1"
