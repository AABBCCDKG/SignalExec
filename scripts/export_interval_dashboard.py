"""Export merged locked-test artifacts into the static dashboard JSON."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


INTERVALS = ("1m", "3m", "5m", "15m", "30m")
RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)
PERIOD_SCOPE = "Training + validation: Jun 2023–Dec 2025\nTest: May–Aug 2026"
METHODS = (
    ("hail_baseline", "rule_based", "Rule-Based Baseline"),
    ("sac", "sac", "SAC"),
    ("ppo", "ppo", "PPO"),
    ("memory_ppo", "memory_ppo", "Memory-PPO"),
)
DESCRIPTIONS = {
    "sac": "Off-policy actor-critic trained with replayed execution transitions.",
    "ppo": "On-policy actor-critic trained with clipped policy updates.",
    "memory_ppo": "Recurrent PPO carrying an LSTM state through each execution day.",
}
CONCLUSIONS = {
    "1m": "PPO leads on mean; Rule-Based and SAC trail TWAP.",
    "3m": "PPO leads on mean but remains seed-sensitive.",
    "5m": "Rule-Based is positive; Memory-PPO is driven by one seed.",
    "15m": "Rule-Based and PPO beat TWAP; PPO remains dispersed.",
    "30m": "Rule-Based leads; SAC turns positive but remains dispersed.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series_for(
    rows: list[dict[str, str]], algorithm: str, public_id: str, label: str
) -> dict:
    selected = [row for row in rows if row["algorithm"] == algorithm]
    seeds = sorted(
        {row["train_seed"] for row in selected},
        key=lambda value: -1 if value == "" else int(value),
    )
    seed_values = [
        [
            float(
                next(
                    row["signal_increment_bps"]
                    for row in selected
                    if row["train_seed"] == seed
                    and float(row["signal_accuracy"]) == rho
                )
            )
            for rho in RHOS
        ]
        for seed in seeds
    ]
    result = {
        "id": public_id,
        "label": label,
        "mean": [mean(values) for values in zip(*seed_values)],
        "seeds": seed_values,
    }
    if algorithm in DESCRIPTIONS:
        result["description"] = DESCRIPTIONS[algorithm]
    return result


def method_for(
    rows: list[dict[str, str]], algorithm: str, public_id: str, label: str
) -> dict:
    values = [
        float(row["twap_shortfall_advantage_bps"])
        for row in rows
        if row["algorithm"] == algorithm
        and row["signal_mode"] == "controlled"
        and float(row["signal_accuracy"]) == 0.8
    ]
    return {"id": public_id, "label": label, "mean": mean(values), "seeds": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=Path("dashboard/data/experiments.json"),
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    dashboard = json.loads(args.dashboard.read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in dashboard["intervals"]}
    for interval in INTERVALS:
        root = args.artifacts / f"interval-matrix-{interval}-locked-test-20261018-v1"
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        gates = metadata["merge_gates"]
        if not (
            metadata["status"] == "complete"
            and gates["all_filled"]
            and gates["signal_hashes_matched"]
            and gates["twap_identity_max_abs_bps"] == 0.0
            and gates["rho_zero_identity_max_abs_bps"] == 0.0
        ):
            raise ValueError(f"locked-test gates did not pass for {interval}")
        did = read_csv(root / "difference_in_differences.csv")
        advantages = read_csv(root / "advantage_summary.csv")
        entry = entries[interval]
        entry.pop("statusLabel", None)
        entry.pop("progress", None)
        entry.update(
            {
                "status": "complete",
                "conclusion": CONCLUSIONS[interval],
                "chartData": {
                    "signalResponse": {
                        "scope": PERIOD_SCOPE,
                        "rhos": list(RHOS),
                        "series": [
                            series_for(did, algorithm, public_id, label)
                            for algorithm, public_id, label in METHODS
                        ],
                    },
                    "vsTwap": {
                        "scope": PERIOD_SCOPE,
                        "methods": [
                            method_for(advantages, algorithm, public_id, label)
                            for algorithm, public_id, label in METHODS
                        ],
                    },
                },
            }
        )

    args.dashboard.write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"updated {args.dashboard} from {len(INTERVALS)} locked-test artifacts")


if __name__ == "__main__":
    main()
