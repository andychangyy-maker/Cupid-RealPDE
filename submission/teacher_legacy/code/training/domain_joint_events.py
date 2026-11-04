"""Deterministic paired event schedules for blocked vs joint-domain training.

This module intentionally has no data-loading or ML-framework dependency.  It
defines the part of the D0/D1 experiment that must be frozen before training:
the exact Sim and Real exposure identities and the order in which they occur.

D0 and D1 consume the same two domain streams.  D0 presents the entire Sim
stream followed by the entire Real stream, while D1 interleaves Sim and Real.
Consequently, a valid paired contract differs only in cross-domain ordering;
the event payloads and within-domain order are exactly identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json


SCHEMA = "b210_domain_joint_event_contract_v1"
EVENT_SCHEMA = "b210_domain_joint_event_v1"
SEED_SCHEMA = "b210_domain_joint_event_seed_v1"
RNG_FINGERPRINT_SCHEMA = "b210_domain_joint_rng_probe_v1"
DOMAINS = ("sim", "real")
ARM_D0 = "d0_blocked"
ARM_D1 = "d1_interleaved"
ARMS = (ARM_D0, ARM_D1)
MAX_SEED = (1 << 63) - 1
_MASK64 = (1 << 64) - 1


def canonical_sha256(value: object) -> str:
    """Return SHA-256 of strict, key-sorted, whitespace-free JSON."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_master_seed(master_seed: int) -> int:
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if not 0 <= master_seed <= MAX_SEED:
        raise ValueError(f"master_seed must be in [0, {MAX_SEED}]")
    return master_seed


def _validate_domain(domain: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    return domain


def _validate_exposure_index(exposure_index: int) -> int:
    if isinstance(exposure_index, bool) or not isinstance(exposure_index, int):
        raise TypeError("exposure_index must be an integer")
    if exposure_index < 0:
        raise ValueError("exposure_index must be non-negative")
    return exposure_index


def _validate_exposures_per_domain(exposures_per_domain: int) -> int:
    if (isinstance(exposures_per_domain, bool)
            or not isinstance(exposures_per_domain, int)):
        raise TypeError("exposures_per_domain must be an integer")
    # At N=1 the two prescribed orders coincide, so it cannot attest an
    # order-only causal comparison.
    if exposures_per_domain < 2:
        raise ValueError("exposures_per_domain must be at least 2")
    return exposures_per_domain


def derive_event_seed(
        master_seed: int, domain: str, exposure_index: int) -> int:
    """Derive a framework-safe 63-bit seed from an event identity."""
    payload = {
        "schema": SEED_SCHEMA,
        "master_seed": _validate_master_seed(master_seed),
        "domain": _validate_domain(domain),
        "exposure_index": _validate_exposure_index(exposure_index),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & MAX_SEED


def _splitmix64(value: int) -> int:
    """One stable SplitMix64 step, used only to fingerprint an RNG stream."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def event_rng_fingerprint(
        master_seed: int, domain: str, exposure_index: int) -> str:
    """Fingerprint a deterministic four-value RNG probe for one event.

    The probe is deliberately implemented here instead of delegating to
    NumPy/PyTorch, whose RNG algorithms can vary with package or device.
    Training code may seed its own RNGs with ``derive_event_seed``; this digest
    provides a library-independent attestation of the intended event stream.
    """
    domain = _validate_domain(domain)
    exposure_index = _validate_exposure_index(exposure_index)
    master_seed = _validate_master_seed(master_seed)
    seed = derive_event_seed(master_seed, domain, exposure_index)
    state = seed
    probe = []
    for _ in range(4):
        state = _splitmix64(state)
        probe.append(f"{state:016x}")
    return canonical_sha256({
        "schema": RNG_FINGERPRINT_SCHEMA,
        "master_seed": master_seed,
        "domain": domain,
        "exposure_index": exposure_index,
        "derived_seed": seed,
        "splitmix64_probe_hex": probe,
    })


def make_event(master_seed: int, domain: str,
               exposure_index: int) -> dict[str, object]:
    """Create the canonical JSON-compatible record for one exposure."""
    master_seed = _validate_master_seed(master_seed)
    domain = _validate_domain(domain)
    exposure_index = _validate_exposure_index(exposure_index)
    return {
        "schema": EVENT_SCHEMA,
        "domain": domain,
        "exposure_index": exposure_index,
        "seed": derive_event_seed(master_seed, domain, exposure_index),
        "rng_fingerprint_sha256": event_rng_fingerprint(
            master_seed, domain, exposure_index),
    }


def generate_domain_event_stream(
        domain: str, exposures_per_domain: int,
        master_seed: int) -> list[dict[str, object]]:
    """Generate one domain's ordered event stream."""
    domain = _validate_domain(domain)
    exposures_per_domain = _validate_exposures_per_domain(
        exposures_per_domain)
    master_seed = _validate_master_seed(master_seed)
    return [
        make_event(master_seed, domain, exposure_index)
        for exposure_index in range(exposures_per_domain)
    ]


def _build_arm_events(
        arm: str,
        sim_events: Sequence[Mapping[str, object]],
        real_events: Sequence[Mapping[str, object]],
        ) -> list[dict[str, object]]:
    # Copy records so mutating one serialized arm cannot silently mutate the
    # other through shared Python object identity.
    sim = [dict(event) for event in sim_events]
    real = [dict(event) for event in real_events]
    if arm == ARM_D0:
        return sim + real
    if arm == ARM_D1:
        return [event for pair in zip(sim, real) for event in pair]
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def _domain_subsequence(
        events: Sequence[Mapping[str, object]], domain: str,
        ) -> list[dict[str, object]]:
    return [dict(event) for event in events if event.get("domain") == domain]


def _contract_payload(contract: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in contract.items()
            if key != "contract_sha256"}


def build_paired_event_contract(
        exposures_per_domain: int, master_seed: int,
        ) -> dict[str, object]:
    """Build a self-attesting D0/D1 order-only paired contract."""
    exposures_per_domain = _validate_exposures_per_domain(
        exposures_per_domain)
    master_seed = _validate_master_seed(master_seed)
    streams = {
        domain: generate_domain_event_stream(
            domain, exposures_per_domain, master_seed)
        for domain in DOMAINS
    }
    d0 = _build_arm_events(ARM_D0, streams["sim"], streams["real"])
    d1 = _build_arm_events(ARM_D1, streams["sim"], streams["real"])
    event_set_sha256 = canonical_sha256(sorted(
        canonical_sha256(event) for event in d0))
    contract: dict[str, object] = {
        "schema": SCHEMA,
        "master_seed": master_seed,
        "exposures_per_domain": exposures_per_domain,
        "event_count_per_arm": 2 * exposures_per_domain,
        "domain_stream_sha256": {
            domain: canonical_sha256(streams[domain]) for domain in DOMAINS
        },
        "event_set_sha256": event_set_sha256,
        "arms": {
            ARM_D0: {
                "schedule": "sim_then_real",
                "events": d0,
                "order_sha256": canonical_sha256(d0),
            },
            ARM_D1: {
                "schedule": "alternating_sim_real",
                "events": d1,
                "order_sha256": canonical_sha256(d1),
            },
        },
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def validate_paired_event_contract(
        contract: Mapping[str, object]) -> dict[str, object]:
    """Validate all digests and the exact order-only D0/D1 invariant.

    Rebuilding from the declared seed and exposure count makes changes to an
    event identity, seed, fingerprint, order, digest, or schedule fail closed.
    The returned report is compact enough to embed in a training audit.
    """
    if not isinstance(contract, Mapping):
        raise TypeError("contract must be a mapping")
    required = {
        "schema", "master_seed", "exposures_per_domain",
        "event_count_per_arm", "domain_stream_sha256", "event_set_sha256",
        "arms", "contract_sha256",
    }
    if set(contract) != required:
        raise ValueError(
            "contract fields differ from schema: "
            f"missing={sorted(required - set(contract))}, "
            f"extra={sorted(set(contract) - required)}")
    if contract["schema"] != SCHEMA:
        raise ValueError(f"unexpected contract schema {contract['schema']!r}")

    expected = build_paired_event_contract(
        _validate_exposures_per_domain(contract["exposures_per_domain"]),
        _validate_master_seed(contract["master_seed"]),
    )
    if contract != expected:
        if contract["contract_sha256"] != canonical_sha256(
                _contract_payload(contract)):
            raise ValueError("contract SHA-256 mismatch")
        raise ValueError("paired event contract differs from canonical schedule")

    arms = contract["arms"]
    assert isinstance(arms, Mapping)  # Guaranteed by equality to ``expected``.
    d0 = arms[ARM_D0]["events"]
    d1 = arms[ARM_D1]["events"]
    within_domain_equal = {
        domain: _domain_subsequence(d0, domain)
        == _domain_subsequence(d1, domain)
        for domain in DOMAINS
    }
    if not all(within_domain_equal.values()):
        raise ValueError("D0/D1 within-domain streams are not identical")
    if sorted(canonical_sha256(event) for event in d0) != sorted(
            canonical_sha256(event) for event in d1):
        raise ValueError("D0/D1 event multisets are not identical")
    if d0 == d1:
        raise ValueError("D0/D1 schedules must differ in cross-domain order")

    return {
        "schema": SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "master_seed": contract["master_seed"],
        "exposures_per_domain": contract["exposures_per_domain"],
        "event_count_per_arm": contract["event_count_per_arm"],
        "same_event_multiset": True,
        "within_domain_streams_equal": within_domain_equal,
        "only_cross_domain_order_differs": True,
        "d0_order_sha256": arms[ARM_D0]["order_sha256"],
        "d1_order_sha256": arms[ARM_D1]["order_sha256"],
    }

