from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "domain_joint_events.py"
SPEC = importlib.util.spec_from_file_location("domain_joint_events", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def arm_events(contract: dict, arm: str) -> list[dict]:
    return contract["arms"][arm]["events"]


def domain_events(events: list[dict], domain: str) -> list[dict]:
    return [event for event in events if event["domain"] == domain]


def test_d0_and_d1_have_same_events_and_only_differ_by_order():
    count = 7
    contract = subject.build_paired_event_contract(count, master_seed=4)
    d0 = arm_events(contract, subject.ARM_D0)
    d1 = arm_events(contract, subject.ARM_D1)

    assert len(d0) == len(d1) == 2 * count
    assert [event["domain"] for event in d0] == ["sim"] * count + [
        "real"] * count
    assert [event["domain"] for event in d1] == [
        domain for _ in range(count) for domain in ("sim", "real")]
    assert d0 != d1
    assert sorted(subject.canonical_sha256(event) for event in d0) == sorted(
        subject.canonical_sha256(event) for event in d1)
    for domain in subject.DOMAINS:
        assert domain_events(d0, domain) == domain_events(d1, domain)
        assert [event["exposure_index"]
                for event in domain_events(d0, domain)] == list(range(count))

    report = subject.validate_paired_event_contract(contract)
    assert report["event_count_per_arm"] == 2 * count
    assert report["same_event_multiset"] is True
    assert report["within_domain_streams_equal"] == {
        "sim": True, "real": True}
    assert report["only_cross_domain_order_differs"] is True
    assert report["d0_order_sha256"] != report["d1_order_sha256"]


def test_event_seed_and_rng_fingerprint_are_identity_deterministic():
    event = subject.make_event(12345, "sim", 9)
    assert event == subject.make_event(12345, "sim", 9)
    assert 0 <= event["seed"] <= subject.MAX_SEED
    assert len(event["rng_fingerprint_sha256"]) == 64

    identities = [
        subject.make_event(12345, "sim", 9),
        subject.make_event(12345, "real", 9),
        subject.make_event(12345, "sim", 10),
        subject.make_event(12346, "sim", 9),
    ]
    assert len({event["seed"] for event in identities}) == len(identities)
    assert len({event["rng_fingerprint_sha256"]
                for event in identities}) == len(identities)


def test_contract_and_canonical_sha_are_reproducible():
    first = subject.build_paired_event_contract(8, master_seed=4)
    second = subject.build_paired_event_contract(8, master_seed=4)
    assert first == second
    assert first["contract_sha256"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "contract_sha256"
    })
    assert subject.canonical_sha256({"b": 2, "a": 1}) == (
        subject.canonical_sha256({"a": 1, "b": 2}))
    with pytest.raises(ValueError):
        subject.canonical_sha256({"invalid": float("nan")})


@pytest.mark.parametrize(
    "tamper",
    [
        lambda value: value["arms"][subject.ARM_D0]["events"][0].__setitem__(
            "exposure_index", 99),
        lambda value: value["arms"][subject.ARM_D1]["events"][1].__setitem__(
            "seed", 0),
        lambda value: value["arms"][subject.ARM_D0]["events"][2].__setitem__(
            "rng_fingerprint_sha256", "0" * 64),
        lambda value: value["arms"][subject.ARM_D1]["events"].reverse(),
        lambda value: value["arms"][subject.ARM_D0].__setitem__(
            "schedule", "alternating_sim_real"),
        lambda value: value.__setitem__("event_set_sha256", "f" * 64),
    ],
)
def test_validator_detects_event_schedule_and_digest_tampering(tamper):
    contract = subject.build_paired_event_contract(5, master_seed=4)
    tamper(contract)
    with pytest.raises(ValueError, match="mismatch|canonical schedule"):
        subject.validate_paired_event_contract(contract)


def test_rehashing_does_not_hide_a_tampered_canonical_schedule():
    contract = subject.build_paired_event_contract(5, master_seed=4)
    events = contract["arms"][subject.ARM_D1]["events"]
    events[0], events[2] = events[2], events[0]
    contract["arms"][subject.ARM_D1]["order_sha256"] = (
        subject.canonical_sha256(events))
    contract["contract_sha256"] = subject.canonical_sha256({
        key: value for key, value in contract.items()
        if key != "contract_sha256"
    })

    with pytest.raises(ValueError, match="canonical schedule"):
        subject.validate_paired_event_contract(contract)


def test_arm_records_do_not_share_mutable_event_objects():
    contract = subject.build_paired_event_contract(3, master_seed=4)
    d0 = arm_events(contract, subject.ARM_D0)
    d1 = arm_events(contract, subject.ARM_D1)
    d0[0]["seed"] = -1
    assert d1[0]["seed"] != -1


@pytest.mark.parametrize("count", [0, 1, -1, True, 2.5])
def test_invalid_exposure_count_is_rejected(count):
    exception = TypeError if isinstance(count, (bool, float)) else ValueError
    with pytest.raises(exception):
        subject.build_paired_event_contract(count, master_seed=4)


def test_unknown_domain_and_boolean_indices_are_rejected():
    with pytest.raises(ValueError, match="domain"):
        subject.make_event(4, "other", 0)
    with pytest.raises(TypeError, match="exposure_index"):
        subject.make_event(4, "sim", True)
    with pytest.raises(TypeError, match="master_seed"):
        subject.make_event(True, "sim", 0)


def test_missing_or_extra_contract_fields_are_rejected():
    missing = subject.build_paired_event_contract(3, master_seed=4)
    missing.pop("event_set_sha256")
    with pytest.raises(ValueError, match="contract fields"):
        subject.validate_paired_event_contract(missing)

    extra = deepcopy(subject.build_paired_event_contract(3, master_seed=4))
    extra["unregistered"] = True
    with pytest.raises(ValueError, match="contract fields"):
        subject.validate_paired_event_contract(extra)
