#!/usr/bin/env python3
"""Generate independent ABR parity fixtures using only Python's standard library.

Run: python3 web/tests/reference.py

The transition formulas are transcribed from processing/abr_env.py;
the clipped logistic policy is from processing/policy.py. The main
configuration follows sections/4_success.tex (eight chunks, seven constant
traces), which differs from the checked-in YAML (four chunks, one trace).

This reference enumerates complete action sequences with itertools.product.
It differentiates the product of trajectory probabilities directly, rather
than calling the browser engine or using its dynamic-programming recurrence.
Reported gradients are of total expected episode return J, not J / horizon.

Arithmetic uses Python float64. Upstream emits float32 observations and uses
PyTorch float32 policy arithmetic; these fixtures validate the literal model,
not bit-for-bit agreement with that lower precision or its random sampler.
The upstream probability clamp [1e-6, 1-1e-6] is retained, including its zero
derivative outside the unclipped interval. No entropy bonus is added to J.
"""

import itertools
import json
import math
from pathlib import Path


BITRATES = (2.85, 4.3)
CHUNK_SECONDS = 4.0
BUFFER_SECONDS = 60.0
EPSILON = 1e-6
MANUSCRIPT = {
    "horizon": 8,
    "bandwidths": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    "lambda": 4.3,
}


def transition(buffer, action, bandwidth, config):
    """Literal drain-then-append dynamics, valid for constant traces."""
    bitrate = BITRATES[action]
    download = bitrate * CHUNK_SECONDS / max(bandwidth, 1e-6)
    stall = max(download - buffer, 0.0)
    remaining = max(buffer - download, 0.0)
    desired = remaining + CHUNK_SECONDS
    if desired > BUFFER_SECONDS:
        remaining -= desired - BUFFER_SECONDS
    next_buffer = remaining + CHUNK_SECONDS
    return {
        "buffer": next_buffer,
        "reward": bitrate / max(BITRATES) - config["lambda"] * stall / CHUNK_SECONDS,
        "stall": stall,
        "bitrate": bitrate,
        "download": download,
    }


def policy(buffer, w, b):
    x = buffer / BUFFER_SECONDS
    z = w * x + b
    if z >= 0:
        raw = 1.0 / (1.0 + math.exp(-z))
    else:
        exponent = math.exp(z)
        raw = exponent / (1.0 + exponent)
    clipped = min(1.0 - EPSILON, max(EPSILON, raw))
    slope = raw * (1.0 - raw) if EPSILON <= raw <= 1 - EPSILON else 0.0
    return clipped, slope * x, slope


def exact(config, w, b):
    totals = {name: 0.0 for name in (
        "return", "gradW", "gradB", "bitrate", "stall", "secondMoment", "probabilityMass"
    )}
    traces = config["bandwidths"]
    horizon = config["horizon"]
    for bandwidth in traces:
        for actions in itertools.product((0, 1), repeat=horizon):
            probability, derivative_w, derivative_b = 1.0, 0.0, 0.0
            buffer, reward, bitrate, stall = 0.0, 0.0, 0.0, 0.0
            for action in actions:
                high, high_w, high_b = policy(buffer, w, b)
                if action:
                    selected, selected_w, selected_b = high, high_w, high_b
                else:
                    selected, selected_w, selected_b = 1.0 - high, -high_w, -high_b
                derivative_w = derivative_w * selected + probability * selected_w
                derivative_b = derivative_b * selected + probability * selected_b
                probability *= selected
                outcome = transition(buffer, action, bandwidth, config)
                buffer = outcome["buffer"]
                reward += outcome["reward"]
                bitrate += outcome["bitrate"]
                stall += outcome["stall"]
            totals["return"] += probability * reward
            totals["gradW"] += derivative_w * reward
            totals["gradB"] += derivative_b * reward
            totals["bitrate"] += probability * bitrate / horizon
            totals["stall"] += probability * stall
            totals["secondMoment"] += probability * reward * reward
            totals["probabilityMass"] += probability
    totals = {name: value / len(traces) for name, value in totals.items()}
    totals["variance"] = max(0.0, totals.pop("secondMoment") - totals["return"] ** 2)
    return totals


def threshold_result(config, threshold):
    """Source oracle convention: choose high at buffer >= threshold."""
    reward, bitrate, stall = 0.0, 0.0, 0.0
    for bandwidth in config["bandwidths"]:
        buffer = 0.0
        for _ in range(config["horizon"]):
            action = int(buffer >= threshold)
            outcome = transition(buffer, action, bandwidth, config)
            buffer = outcome["buffer"]
            reward += outcome["reward"]
            bitrate += outcome["bitrate"]
            stall += outcome["stall"]
    count = len(config["bandwidths"])
    return {
        "return": reward / count,
        "threshold": threshold,
        "bitrate": bitrate / count / config["horizon"],
        "stall": stall / count,
    }


def threshold_oracle(config):
    """Enumerate all threshold-induced behaviors, independently of the JS engine.

    Every reachable pre-action buffer is a boundary. A threshold strictly
    between adjacent boundaries has one constant action mapping. Checking
    these intervals plus the endpoints covers all deterministic thresholds.
    """
    boundaries = {0.0, BUFFER_SECONDS}
    for bandwidth in config["bandwidths"]:
        states = {0.0}
        for _ in range(config["horizon"]):
            boundaries.update(states)
            states = {
                transition(buffer, action, bandwidth, config)["buffer"]
                for buffer in states
                for action in (0, 1)
            }
    ordered = sorted(boundaries)
    thresholds = [0.0, BUFFER_SECONDS] + [
        (left + right) / 2.0
        for left, right in zip(ordered, ordered[1:])
        if right - left > 1e-10
    ]
    best = None
    for threshold in sorted(thresholds):
        outcome = threshold_result(config, threshold)
        if best is None or outcome["return"] > best["return"] + 1e-12:
            best = outcome
    # Retain the original repository's coarser 0..100% threshold search too.
    grid = [threshold_result(config, percent * BUFFER_SECONDS / 100) for percent in range(101)]
    grid_best = max(grid, key=lambda outcome: outcome["return"])
    return {"expected": best, "sourceIntegerPercentGrid": grid_best}


def build_fixtures():
    low_penalty = {**MANUSCRIPT, "lambda": 1.3}
    narrow = {**MANUSCRIPT, "bandwidths": [3.0, 3.5, 4.0]}
    repository = {**MANUSCRIPT, "horizon": 4, "bandwidths": [4.0]}
    cases = [
        ("manuscript_uniform", MANUSCRIPT, 0.0, 0.0),
        ("manuscript_threshold_shape", MANUSCRIPT, 20.0, -2.0),
        ("manuscript_inverted_initialization", MANUSCRIPT, -40.0, 8.0),
        ("manuscript_fully_clipped_high", MANUSCRIPT, 3.0, 20.0),
        ("low_stall_penalty", low_penalty, 25.0, -1.0),
        ("narrow_training_traces", narrow, 10.0, -1.0),
        ("committed_yaml_uniform", repository, 0.0, 0.0),
    ]
    policies = []
    for name, config, w, b in cases:
        expected = exact(config, w, b)
        assert abs(expected["probabilityMass"] - 1.0) < 1e-10
        # Check the independent analytic product derivatives numerically.
        h = 1e-4
        finite_w = (exact(config, w + h, b)["return"] - exact(config, w - h, b)["return"]) / (2 * h)
        finite_b = (exact(config, w, b + h)["return"] - exact(config, w, b - h)["return"]) / (2 * h)
        assert abs(expected["gradW"] - finite_w) < 1e-7, name
        assert abs(expected["gradB"] - finite_b) < 1e-7, name
        policies.append({"name": name, "config": config, "params": {"w": w, "b": b}, "expected": expected})
    transitions = []
    for name, buffer, action, bandwidth, penalty in [
        ("startup_low", 0.0, 0, 4.0, 4.3),
        ("startup_high", 0.0, 1, 6.0, 4.3),
        ("partial_rebuffer", 5.0, 1, 3.0, 4.3),
        ("exactly_covers_download", 4.3, 1, 4.0, 4.3),
        ("drain_at_full_buffer", 59.0, 0, 20.0, 4.3),
        ("low_stall_penalty", 0.0, 1, 3.0, 1.3),
    ]:
        config = {"lambda": penalty}
        transitions.append({
            "name": name,
            "input": {"buffer": buffer, "action": action, "bandwidth": bandwidth, "config": config},
            "expected": transition(buffer, action, bandwidth, config),
        })
    baselines = []
    for name, config in [("manuscript", MANUSCRIPT), ("low_penalty", low_penalty), ("narrow_traces", narrow), ("committed_yaml", repository)]:
        baselines.append({"name": name, "config": config, **threshold_oracle(config)})
    return {
        "provenance": {
            "generator": "web/tests/reference.py; Python standard library only",
            "sources": ["processing/abr_env.py", "processing/policy.py", "workspace manuscript: sections/4_success.tex", "processing/oracle_multitrace.py"],
            "precision": "float64 literal model; upstream observations and policy arithmetic use float32",
            "probabilityClamp": [EPSILON, 1.0 - EPSILON],
            "gradient": "d(total expected episode reward)/d(w,b), without horizon scaling or entropy regularization",
            "baseline": "Best deterministic single buffer threshold over all reachable-state intervals; threshold in seconds. Source integer-percent grid is also recorded.",
        },
        "transitions": transitions,
        "policies": policies,
        "baselines": baselines,
    }


if __name__ == "__main__":
    destination = Path(__file__).with_name("reference.json")
    destination.write_text(json.dumps(build_fixtures(), indent=2) + "\n")
    print(f"Generated {destination}")
