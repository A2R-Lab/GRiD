"""A cached relink must not hide a sibling variant's measured compile peak."""
import sys

import pytest

sys.path.insert(0, "test")
import compile_sched


@pytest.fixture
def ledger(tmp_path):
    result = compile_sched.Ledger(tmp_path / "rss.json")
    result.record("warm:iiwa14,mb=64", 700_000)
    result.record("warm:iiwa14,mb=1024", 14_000_000)
    result.record("warm:iiwa140,mb=64", 30_000_000)
    result.record("prewarm:iiwa14", 40_000_000)
    return result


@pytest.mark.parametrize("key", ["warm:iiwa14,mb=64", "warm:iiwa14,mb=4096", "warm:iiwa14"])
def test_warm_prediction_uses_robot_peak(ledger, key):
    assert ledger.predict_kb(key, 8_000_000, 1.3) == 18_200_000


def test_unmeasured_variant_keeps_larger_default(ledger):
    assert ledger.predict_kb("warm:iiwa14,new=1", 20_000_000, 1.3) == 26_000_000


def test_other_robots_and_non_warm_keys_are_not_grouped(ledger):
    assert ledger.predict_kb("warm:other,mb=64", 8_000_000, 1) == 8_000_000
    assert ledger.predict_kb("prewarm:iiwa14", 8_000_000, 1) == 40_000_000
    assert ledger.predict_kb("prewarm:iiwa14,new=1", 8_000_000, 1) == 8_000_000


def test_floor_survives_reload_and_small_relink(ledger):
    ledger.record("warm:iiwa14,mb=1024", 1)
    reloaded = compile_sched.Ledger(ledger.path)
    assert reloaded.predict_kb("warm:iiwa14,mb=64", 8_000_000, 1) == 14_000_000


def test_first_job_admission_is_still_an_explicit_exception(ledger, monkeypatch):
    # Preserve the existing progress policy: an empty pool admits one job even
    # if its estimate exceeds the available budget. The new floor is NOT an OOM
    # guarantee; the operator still needs a feasible single-job compile profile.
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: 1_000_000)
    scheduler = compile_sched.RamScheduler(ledger, floor_kb=1_000_000, default_peak_kb=1_000_000)
    observed = []

    def launch(job, predicted, running, groups):
        observed.append(predicted)
        scheduler.stop()
        return False

    monkeypatch.setattr(scheduler, "_launch", launch)
    monkeypatch.setattr(compile_sched.time, "sleep", lambda _: None)
    job = compile_sched.Job("first", ["not-executed"], "warm:iiwa14,mb=64")
    scheduler.run([job])
    assert observed == [18_200_000]
    assert scheduler.results["first"].rc == -1


def test_historical_small_relinks_cannot_all_fit_the_jetson_budget(ledger):
    # 61 GiB total is not the available budget: reserve the 10 GiB floor
    # against a plausible 56 GiB MemAvailable snapshot. Three iiwa14 builds
    # previously looked like 14 + .7 + .7 GiB, plus go2.
    gib = 1024 ** 2
    ledger.record("warm:iiwa14,mb=8", 14 * gib)
    ledger.record("warm:go2,mb=8", 14 * gib)
    budget = (56 - 10) * gib
    keys = ["warm:iiwa14,mb=8", "warm:go2,mb=8", "warm:iiwa14,mb=64", "warm:iiwa14,mb=256"]
    peaks = [ledger.predict_kb(key, 8 * gib, 1.3) for key in keys]
    assert sum(peaks[:2]) <= budget < sum(peaks[:3])
