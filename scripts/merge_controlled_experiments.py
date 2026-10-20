"""Merge independently trained controlled-signal artifacts and re-run gates."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path

from hail_ppo_reproduction.controlled_experiment import (
    difference_in_differences,
    paired_advantages,
    plot_signal_increment,
    summarize_advantages,
    summarize_controlled,
    validate_matched_records,
)
from hail_ppo_reproduction.experiment import ALGORITHMS, CONTROLS


FLOAT_FIELDS = {
    "signal_accuracy",
    "pnl_bps",
    "implementation_shortfall_bps",
    "average_execution_price",
    "fill_fraction",
    "episode_reward",
    "cost_bps",
}
INT_FIELDS = {"target_direction", "signal_replica"}


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            record = dict(raw)
            record["train_seed"] = (
                None if record["train_seed"] == "" else int(record["train_seed"])
            )
            for field in FLOAT_FIELDS:
                record[field] = (
                    None if record[field] == "" else float(record[field])
                )
            for field in INT_FIELDS:
                record[field] = int(record[field])
            record["filled"] = record["filled"].lower() == "true"
            records.append(record)
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls-source", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--before-date",
        type=date.fromisoformat,
        help="keep only episodes with date strictly before this boundary",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    control_metadata = json.loads(
        (args.controls_source / "metadata.json").read_text(encoding="utf-8")
    )
    control_records = [
        record
        for record in read_records(args.controls_source / "episodes.csv")
        if record["algorithm"] in CONTROLS
    ]
    if {record["algorithm"] for record in control_records} != set(CONTROLS):
        raise ValueError("controls source does not contain every registered control")

    learned_records = []
    run_metadata = []
    combinations = set()
    for run in args.runs:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            raise ValueError(f"run is not complete: {run}")
        if metadata["evaluation_splits"] != control_metadata["evaluation_splits"]:
            raise ValueError(f"split mismatch in {run}")
        if metadata["historical_data"] != control_metadata["historical_data"]:
            raise ValueError(f"historical-data mismatch in {run}")
        if len(metadata["algorithms"]) != 1 or len(metadata["experiment"]["train_seeds"]) != 1:
            raise ValueError(f"run must contain exactly one algorithm and seed: {run}")
        combination = (
            metadata["algorithms"][0],
            metadata["experiment"]["train_seeds"][0],
        )
        if combination in combinations:
            raise ValueError(f"duplicate algorithm/seed run: {combination}")
        combinations.add(combination)
        learned_records.extend(
            record
            for record in read_records(run / "episodes.csv")
            if record["algorithm"] in ALGORITHMS
        )
        run_metadata.append(metadata)

    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in (0, 1, 2)}
    if combinations != expected:
        raise ValueError(f"expected algorithm/seed matrix {expected}, found {combinations}")

    records = control_records + learned_records
    if args.before_date is not None:
        boundary = args.before_date.isoformat()
        records = [record for record in records if record["date"] < boundary]
    identities = {
        (record["algorithm"], record["train_seed"], record["matched_input_id"])
        for record in records
    }
    if len(identities) != len(records):
        raise ValueError("merged records contain duplicate policy/input rows")
    validate_matched_records(records)

    summaries = summarize_controlled(records)
    advantages = paired_advantages(records)
    advantage_summaries = summarize_advantages(advantages)
    did = difference_in_differences(advantages)
    twap_error = max(
        abs(record["twap_shortfall_advantage_bps"])
        for record in advantages
        if record["algorithm"] == "twap"
    )
    rho_zero_error = max(
        abs(record["signal_increment_bps"])
        for record in did
        if record["signal_accuracy"] == 0.0
    )
    if twap_error > 1e-12 or rho_zero_error > 1e-12:
        raise ValueError(
            f"comparison identity failed: twap={twap_error}, rho_zero={rho_zero_error}"
        )

    args.output.mkdir(parents=True)
    write_csv(args.output / "episodes.csv", records)
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "paired_advantages.csv", advantages)
    write_csv(args.output / "advantage_summary.csv", advantage_summaries)
    write_csv(args.output / "difference_in_differences.csv", did)
    (args.output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "difference_in_differences.json").write_text(
        json.dumps(did, indent=2) + "\n", encoding="utf-8"
    )
    plot_signal_increment(did, args.output / "signal_increment.png")

    policy_count = len(
        {(record["algorithm"], record["train_seed"]) for record in records}
    )
    matched_input_count = len({record["matched_input_id"] for record in records})
    metadata = {
        **control_metadata,
        "status": "complete",
        "name": "Signal-to-Execution RL: merged parallel validation",
        "controls_source": str(args.controls_source),
        "run_sources": [str(run) for run in args.runs],
        "merged_before_date": (
            None if args.before_date is None else args.before_date.isoformat()
        ),
        "algorithms": list(ALGORITHMS),
        "controls": list(CONTROLS),
        "training": [
            training
            for metadata in run_metadata
            for training in metadata["training"]
        ],
        "merge_gates": {
            "records": len(records),
            "policies": policy_count,
            "matched_inputs": matched_input_count,
            "all_filled": all(record["filled"] for record in records),
            "signal_hashes_matched": True,
            "twap_identity_max_abs_bps": twap_error,
            "rho_zero_identity_max_abs_bps": rho_zero_error,
        },
    }
    metadata.pop("error", None)
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"complete: {len(records)} records, {policy_count} policies, "
        f"{matched_input_count} matched inputs -> {args.output}"
    )


if __name__ == "__main__":
    main()
