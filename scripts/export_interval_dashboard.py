"""Export merged locked-test artifacts into the static dashboard JSON."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


INTERVALS = ("1m", "3m", "5m", "15m", "30m")
RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)
METHODS = (
    ("hail_baseline", "HAIL baseline"),
    ("sac", "SAC"),
    ("ppo", "PPO"),
    ("memory_ppo", "Memory-PPO"),
)
CONCLUSIONS = {
    "1m": "PPO leads on mean; HAIL and SAC trail TWAP.",
    "3m": "PPO leads on mean but remains seed-sensitive.",
    "5m": "HAIL is positive; Memory-PPO is driven by one seed.",
    "15m": "HAIL and PPO beat TWAP; PPO remains dispersed.",
    "30m": "HAIL leads; SAC turns positive but remains dispersed.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def series_for(rows: list[dict[str, str]], algorithm: str, label: str) -> dict:
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
    return {
        "id": algorithm,
        "label": label,
        "mean": [mean(values) for values in zip(*seed_values)],
        "seeds": seed_values,
    }


def method_for(rows: list[dict[str, str]], algorithm: str, label: str) -> dict:
    values = [
        float(row["twap_shortfall_advantage_bps"])
        for row in rows
        if row["algorithm"] == algorithm
        and row["signal_mode"] == "controlled"
        and float(row["signal_accuracy"]) == 0.8
    ]
    return {"id": algorithm, "label": label, "mean": mean(values), "seeds": values}


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
        entry.update(
            {
                "status": "complete",
                "statusLabel": "5 symbols · Jun 2023–Aug 2026 · complete",
                "progress": 100.0,
                "conclusion": CONCLUSIONS[interval],
                "chartData": {
                    "signalResponse": {
                        "scope": "History Jun 2023–Aug 2026 · locked test May–Aug 2026",
                        "rhos": list(RHOS),
                        "series": [
                            series_for(did, algorithm, label)
                            for algorithm, label in METHODS
                        ],
                    },
                    "vsTwap": {
                        "scope": "History Jun 2023–Aug 2026 · locked test rho=0.8",
                        "methods": [
                            method_for(advantages, algorithm, label)
                            for algorithm, label in METHODS
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
