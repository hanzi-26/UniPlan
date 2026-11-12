#!/usr/bin/env python3
"""Run the four published workloads with UniPlan and its comparison methods."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from experiments.artifacts import bind_json, read


ROOT = Path(__file__).resolve().parent
WORKLOADS = ("arxiv", "spider", "wiki", "yfcc")
MANIFEST = ROOT / "data/published-workloads.json"
BASELINES = ROOT / "configs/main/baselines.json"
DEFAULT_ASSETS = ROOT / "assets"


def verify_workload(workload, hashes):
    for name in ("queries", "splits", "source_sql"):
        if name not in workload:
            continue
        filename = workload[name]
        digest = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if digest != hashes[filename]:
            raise ValueError(f"Published workload checksum changed: {filename}")
    split = read(ROOT / workload["splits"])
    if {role: len(ids) for role, ids in split.items()} != workload["split_counts"]:
        raise ValueError("Published split counts changed")
    ids = [qid for members in split.values() for qid in members]
    if len(ids) != len(set(ids)):
        raise ValueError("Query IDs overlap across splits")
    return split


def configure(kind, workload, output, args):
    template = args.config_dir / f"{kind}.json"
    placeholders = {
        "@ASSETS@": str(args.assets_dir),
        "@OUTPUT@": str(output),
        "@DATABASE@": str(args.database_config),
    }
    # Substitute inside JSON strings without breaking quotes in a local path.
    raw = template.read_text()
    for token, value in placeholders.items():
        raw = raw.replace(token, json.dumps(value)[1:-1])
    config = json.loads(raw)
    budget = args.budget_ms or (20000 if kind == "yfcc" else 5000)
    config["reward"].update(recall_threshold=args.recall, latency_cap_ms=budget)
    config["training_latency_cap_ms"] = budget
    config["runtime"]["statement_timeout_ms"] = budget
    config["quality"]["required_hits"] = {
        str(k): math.ceil(args.recall * k - 1e-12)
        for k in config["quality"]["allowed_k"]
    }
    config["max_ucb"]["quality_probability_min"] = args.probability
    if args.training_gpus:
        config["runtime"]["main_training_gpus"] = args.training_gpus

    dataset = {
        "kind": kind,
        "queries": str(ROOT / workload["queries"]),
        "splits": str(ROOT / workload["splits"]),
        "query_count": workload["source_query_count"],
        "experiment_config": str(output / "experiment.json"),
        "baselines": str(output / "baselines.json"),
        "output_dir": str(output),
        "seed": 42,
        "repeats": 5,
    }
    calibration = output / "dynamicann-calibration"
    if kind != "yfcc":
        dataset["dynamicann_setup"] = {
            "kind": kind, "output": str(calibration),
            "resources": str(output / "resources.json"),
        }
    if kind == "arxiv":
        dataset["checkpoint"] = str(output / "train/frozen_model")
    elif kind == "wiki":
        # The adapter writes parsed queries and the split into the run directory.
        dataset.update(source=str(ROOT / workload["source_sql"]),
                       queries=str(output / "query-data/queries.jsonl"),
                       splits=str(output / "query-data/splits.json"),
                       split_protocol="provided-semantic-grouped-v1",
                       experiment_role="main")
    elif kind == "spider":
        asset_dir = args.assets_dir / "spider"
        dataset.update(source=str(ROOT / workload["source_sql"]),
                       metadata=str(asset_dir / "find_semantic_tables.json"),
                       catalog=str(asset_dir / "indexes/catalog.json"),
                       schema=str(ROOT / "data/spider/database-columns-v1.json"),
                       allowed_missing_queries=[])
    else:
        dataset["source"] = dataset["queries"]
        calibration = output / "all-methods"

    baselines = read(BASELINES)
    if args.without_jev:
        baselines["baselines"] = [b for b in baselines["baselines"] if b["id"] != "jev_router"]
    for spec in baselines["baselines"]:
        options = spec["config"]
        if spec["id"] == "dynamic_ann":
            options.update(density_artifact=str(calibration / "density.json"),
                           calibration_artifact=str(calibration / (
                               "dynamicann-calibrated.json" if kind == "yfcc"
                               else "dynamicann_calibrated_measured.json")))
        elif spec["name"] == "jev-router":
            options.update(recall_threshold=args.recall,
                           base_url=os.environ.get("TYPESAFE_BASE_URL") or options["base_url"])
        elif kind in ("wiki", "spider") and spec["name"].startswith("pure-"):
            path_id = spec["name"].removeprefix("pure-").upper()
            spec.update(name="fixed-wiki", config={"path_id": path_id})
    dataset["evaluation_methods"] = ["cash_maxucb", *(b["id"] for b in baselines["baselines"])]
    return config, dataset, baselines


def run_workload(kind, output, config, dataset, baselines, split, assets):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reproduce.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind_json(output / "experiment.json", config)
        bind_json(output / "dataset.json", dataset)
        bind_json(output / "baselines.json", baselines)
        if kind == "wiki":
            bind_json(Path(dataset["splits"]), split)
        if kind == "yfcc":
            from experiments.yfcc import run_yfcc

            run_yfcc(output, config, dataset, baselines, split, assets)
            return
        if kind == "arxiv":
            command = ["-m", "experiments.arxiv", "all"]
        else:
            command = ["-m", "experiments.relational", "--kind", kind]
        subprocess.run(
            [sys.executable, "-u", *command, "--dataset", str(output / "dataset.json")],
            cwd=ROOT, check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run"))
    parser.add_argument("--dataset", choices=("all", *WORKLOADS), default="all")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--database-config", type=Path, default=ROOT / "configs/database.yaml")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/main")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/main")
    parser.add_argument("--recall", type=float, default=0.8)
    parser.add_argument("--probability", type=float, default=0.9)
    parser.add_argument("--budget-ms", type=int)
    parser.add_argument("--training-gpus", nargs="+", metavar="DEVICE")
    parser.add_argument("--without-jev", action="store_true", help="Run only the nine local baselines")
    args = parser.parse_args()
    if not 0 < args.recall <= 1 or not 0 < args.probability <= 1:
        parser.error("recall and probability must be in (0, 1]")
    if args.budget_ms is not None and args.budget_ms <= 0:
        parser.error("--budget-ms must be positive")
    for name in ("assets_dir", "database_config", "config_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.stage == "run":
        if not args.database_config.is_file() or not args.assets_dir.is_dir():
            parser.error("set --database-config and --assets-dir to the prepared backend assets")
        if not args.without_jev and (
            not os.environ.get("TYPESAFE_API_KEY") or importlib.util.find_spec("typesafe_sdk") is None
        ):
            parser.error("full comparison requires typesafe-sdk and TYPESAFE_API_KEY")
    source = str(ROOT / "src")
    sys.path.insert(0, source)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH"))))
    manifest = read(MANIFEST)
    for kind in WORKLOADS if args.dataset == "all" else (args.dataset,):
        workload = manifest["datasets"][kind]
        split = verify_workload(workload, manifest["sha256"])
        suffix = "-local" if args.without_jev else ""
        budget = args.budget_ms or (20000 if kind == "yfcc" else 5000)
        output = args.output_dir / f"{kind}-r{args.recall:g}-p{args.probability:g}-{budget}ms{suffix}"
        config, dataset, baselines = configure(kind, workload, output, args)
        if args.stage == "plan":
            print(json.dumps({
                "dataset": kind, "split_counts": workload["split_counts"],
                "methods": dataset["evaluation_methods"], "recall": args.recall,
                "probability": args.probability, "budget_ms": budget,
                "output": str(output),
            }))
        else:
            run_workload(kind, output, config, dataset, baselines, split, args.assets_dir)


if __name__ == "__main__":
    main()
