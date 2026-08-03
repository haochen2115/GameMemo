from gamememo.commons import run_commons


def test_commons_fixture_exposes_the_intended_tradeoff():
    results = {row.policy: row for row in run_commons()}
    assert results["no_memory"].experience_gain == 0.0
    assert results["pooled_raw"].instance_leakage > 0.0
    assert results["pooled_raw"].pseudo_generalization > 0.0
    assert results["quorum"].experience_gain == 1.0
    assert results["quorum"].instance_leakage == 0.0
    assert results["quorum"].pseudo_generalization == 0.0
