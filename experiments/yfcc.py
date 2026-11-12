"""YFCC1920 preparation, training, and benchmark workflow."""

import os

from experiments.artifacts import bind_json


def run_yfcc(output, config, dataset, baselines, split, assets):
    # The corpus path must be configured before importing the YFCC readers.
    os.environ["YFCC_ASSET_DIR"] = str(assets / "yfcc")
    from vec2plan_bandit.surrogates.parallel_fit import shutdown_training_workers
    from vec2plan_bandit.yfcc.all_methods import AllMethodsExperiment, LABELS
    from vec2plan_bandit.yfcc.assets import exact_truth
    from vec2plan_bandit.yfcc.data import load_queries, validate_splits

    queries = load_queries(dataset["source"], metric="l2", expected_count=dataset["query_count"])
    validate_splits(queries, split)
    bind_json(output / "splits.json", split)
    labels = {key: LABELS[key] for key in dataset["evaluation_methods"]}
    extra = output / "all-methods"
    # Fixed paths and native PG execute directly. Only fitted methods enter the model registry.
    fitted = [b for b in baselines["baselines"] if b["id"] in {
        "maxucb_bo", "mlp", "lightgbm", "dynamic_ann", "jev_router"
    }]
    bind_json(extra / "baselines.json", {**baselines, "baselines": fitted})
    bind_json(extra / "contract.json", {
        "labels": labels, "density_sample_size": 2048, "density_clusters": 16,
        "dynamicann_train_queries": 30, "dynamicann_repeats": 3,
        "dynamicann_offline_budget_ms": config["reward"]["latency_cap_ms"],
    })
    receipt = output / "truth.complete.json"
    if not receipt.exists():
        exact_truth(output, dataset["source"], metric="l2", expected_count=len(queries))
        bind_json(receipt, {"complete": True})
    run = AllMethodsExperiment(output)
    try:
        for stage, action in (
            ("plans", run.plans), ("collect", run.collect), ("train", run.train),
            ("calibrate", run.calibrate_final), ("baseline_setup", run.baseline_setup),
            ("dev", lambda: run.evaluate("dev")), ("test", lambda: run.evaluate("test")),
        ):
            marker = output / f"reproduce-{stage}.complete.json"
            if not marker.exists():
                print(f"YFCC: {stage}", flush=True)
                action()
                if stage == "train":
                    shutdown_training_workers()
                bind_json(marker, {"complete": True})
    finally:
        shutdown_training_workers()
        if run.worker is not None:
            run.worker.close()
