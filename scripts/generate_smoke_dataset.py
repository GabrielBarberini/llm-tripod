"""
Generate a synthetic IoT dataset + RAG documents for end-to-end smoke testing.

Outputs (under --out-dir):
- rag_docs.jsonl: documents the RAG leg will retrieve (policies/heuristics).
- train.jsonl / test.jsonl: supervised examples for LoRA training and evaluation.

Each example contains:
- domain: string
- policy_id: string (points to a policy doc in RAG)
- sensor_data: {temp, vibration, status}
- expected: {action, parameters, reasoning}

The key property: policy values (fan/voltage) live in RAG docs. Test samples
can optionally use policy_ids that are not present in the training split (policy
holdout), so correct parameter selection requires RAG context (not memorization).
"""

from __future__ import annotations

import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STATUSES = ["stable", "warning", "critical"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="training_data/smoke", help="Output directory")
    p.add_argument("--n", type=int, default=2000, help="Number of samples")
    p.add_argument("--num-policies", type=int, default=50, help="Number of distinct policy docs")
    p.add_argument("--train-policy-ratio", type=float, default=0.8, help="Fraction of policies used for training split")
    p.add_argument(
        "--holdout-policies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, test split uses policy_ids not present in train (forces RAG).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    return p.parse_args()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def make_policy(rng: random.Random, policy_id: str) -> Dict:
    warning_fan = rng.randint(70, 90)
    critical_fan = rng.randint(min(warning_fan + 5, 95), 100)
    vb_warning = round(rng.uniform(-0.12, -0.02), 2)
    return {
        "policy_id": policy_id,
        "warning_fan": int(warning_fan),
        "critical_fan": int(critical_fan),
        "voltage_bias_warning": float(vb_warning),
        "voltage_bias_critical": float(clamp(vb_warning - 0.10, -0.30, 0.0)),
    }


def decide(policy: Dict, sensor: Dict) -> Tuple[str, Dict, str]:
    temp = float(sensor["temp"])
    vibration = float(sensor["vibration"])
    status = str(sensor["status"]).lower()

    if status == "critical" or temp >= 90:
        action = "set_thermal_profile"
        parameters = {
            "fan_speed": int(policy["critical_fan"]),
            "voltage_bias": float(policy["voltage_bias_critical"]),
        }
        reasoning = "Critical thermal risk: maximize cooling and reduce voltage."
        return action, parameters, reasoning

    if vibration >= 1.0:
        action = "schedule_maintenance"
        parameters = {"fan_speed": int(70), "voltage_bias": float(-0.05)}
        reasoning = "High vibration suggests mechanical risk: schedule maintenance."
        return action, parameters, reasoning

    if status == "warning" or temp >= 80:
        action = "set_thermal_profile"
        parameters = {
            "fan_speed": int(policy["warning_fan"]),
            "voltage_bias": float(policy["voltage_bias_warning"]),
        }
        reasoning = "Warning state: apply profile policy for cooling vs power trade-off."
        return action, parameters, reasoning

    action = "maintain"
    parameters = {"fan_speed": int(60), "voltage_bias": float(0.0)}
    reasoning = "Stable state: maintain baseline settings."
    return action, parameters, reasoning


def make_sensor(rng: random.Random) -> Dict:
    # Bias toward warning/critical to ensure policy parameters matter.
    status = rng.choices(STATUSES, weights=[0.35, 0.45, 0.20])[0]
    if status == "stable":
        temp = rng.uniform(55, 79)
    elif status == "warning":
        temp = rng.uniform(75, 92)
    else:
        temp = rng.uniform(88, 105)

    vibration = rng.uniform(0.05, 1.4)
    # couple vibration with status a bit
    if status == "critical":
        vibration = clamp(vibration + rng.uniform(0.0, 0.3), 0.05, 1.4)
    return {"temp": round(temp, 1), "vibration": round(vibration, 2), "status": status}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Policies: generate a set of distinct policy docs.
    policies = [make_policy(rng, f"policy_{i:03d}") for i in range(args.num_policies)]
    policy_ids = [p["policy_id"] for p in policies]
    rng.shuffle(policy_ids)

    if args.holdout_policies:
        train_policy_count = max(1, int(len(policy_ids) * args.train_policy_ratio))
        train_policy_ids = set(policy_ids[:train_policy_count])
        test_policy_ids = set(policy_ids[train_policy_count:]) or set(policy_ids[:1])
    else:
        train_policy_ids = set(policy_ids)
        test_policy_ids = set(policy_ids)

    rag_docs = []
    for p in policies:
        rag_docs.append(
            {
                "id": p["policy_id"],
                "type": "policy",
                "policy_id": p["policy_id"],
                "text": (
                    f"Policy {p['policy_id']}: "
                    f"In WARNING set fan_speed={p['warning_fan']} and voltage_bias={p['voltage_bias_warning']}. "
                    f"In CRITICAL set fan_speed={p['critical_fan']} and voltage_bias={p['voltage_bias_critical']}."
                ),
            }
        )

    # Generic heuristics that apply regardless of policy_id.
    rag_docs.extend(
        [
            {
                "id": "heuristic_vibration",
                "type": "heuristic",
                "text": "If vibration >= 1.0g, schedule maintenance and reduce voltage slightly (around -0.05).",
            },
            {
                "id": "heuristic_stable",
                "type": "heuristic",
                "text": "If status is stable and temp < 80C, maintain baseline settings.",
            },
        ]
    )
    write_jsonl(out_dir / "rag_docs.jsonl", rag_docs)

    # Generate train/test samples with policy-id holdout:
    # train samples only use train_policy_ids; test samples only use test_policy_ids.
    n_train = int(args.n * (1.0 - args.test_ratio))
    n_test = max(1, args.n - n_train)

    policy_by_id = {p["policy_id"]: p for p in policies}
    train_policy_list = sorted(train_policy_ids)
    test_policy_list = sorted(test_policy_ids)

    train = []
    for _ in range(n_train):
        policy_id = rng.choice(train_policy_list)
        sensor = make_sensor(rng)
        action, parameters, reasoning = decide(policy_by_id[policy_id], sensor)
        train.append(
            {
                "domain": "Thermal Control",
                "policy_id": policy_id,
                "sensor_data": sensor,
                "expected": {"action": action, "parameters": parameters, "reasoning": reasoning},
            }
        )

    test = []
    for _ in range(n_test):
        policy_id = rng.choice(test_policy_list)
        sensor = make_sensor(rng)
        action, parameters, reasoning = decide(policy_by_id[policy_id], sensor)
        test.append(
            {
                "domain": "Thermal Control",
                "policy_id": policy_id,
                "sensor_data": sensor,
                "expected": {"action": action, "parameters": parameters, "reasoning": reasoning},
            }
        )

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "test.jsonl", test)
    print(
        f"Wrote {len(train)} train and {len(test)} test samples to {out_dir} "
        f"(train policies={len(train_policy_ids)}, test policies={len(test_policy_ids)}, holdout={args.holdout_policies})"
    )


if __name__ == "__main__":
    main()
