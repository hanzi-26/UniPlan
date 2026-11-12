# UniPlan

**UniPlan: Optimizing the Performance and Recall of Filtered Vector Searches via A Unified Framework**

UniPlan jointly selects a filtering strategy, relational execution settings, and vector index parameters. It trains the configuration selector offline and uses a calibrated direct execution route for queries that can skip joint selection.

## Run the main comparison

Use Python 3.11 or newer and install the experiment dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
python reproduce.py plan
```

`plan` checks the published workload files and prints each dataset's splits, methods, and settings. It requires no database or model service. After preparing the backends described below, run:

```bash
python reproduce.py run --assets-dir /path/to/assets --database-config configs/database.yaml
```

The default comparison runs UniPlan and ten baselines: PRE, POST, FAISS (PURE), ACORN (INLINE), PostgreSQL/pgvector, MaxUCB+BO, MLP, LightGBM, Dynamic ANN, and JEV. Set `TYPESAFE_API_KEY` in your shell for the JEV baseline. Use `--without-jev` for a local comparison with the other nine baselines.

Use `--dataset arxiv`, `spider`, `wiki`, or `yfcc` to run a single workload. The defaults are recall 0.8, predicted quality threshold 0.9, one warmup, and five measured repeats. The end-to-end budget is 5 seconds for arXiv, Spider, and Wikipedia and 20 seconds for YFCC. Use `--recall`, `--probability`, and `--budget-ms` to select another experiment setting. Offline fitting runs on CPU by default. On a host with compatible GPUs, use `--training-gpus 0 1` to fit on those devices.

The runner prepares exact references, collects training measurements, fits the models, calibrates the routing policy and Dynamic ANN, and evaluates the frozen methods on dev and test. The default output directory is `outputs/main/`. Each run directory contains its resolved configuration, per-request JSONL records, and summaries. Rerun the same command to resume an interrupted run, or use `--output-dir` for a fresh run.

## Repository layout

| Path | Contents |
| --- | --- |
| [`reproduce.py`](reproduce.py) | Entry point for all four workloads |
| [`experiments/`](experiments) | arXiv and YFCC workflows, the shared Wikipedia and Spider workflow, and Dynamic ANN calibration |
| [`src/vec2plan_bandit/`](src/vec2plan_bandit) | UniPlan and baseline implementations, dataset adapters, and backend execution |
| [`configs/main/`](configs/main) | Workload and baseline settings |
| [`data/`](data) | Published queries and splits |
| [`scripts/`](scripts) and [`services/`](services) | Index setup and embedding services |

## Prepare the backends

Copy [the database template](configs/database.example.yaml) to `configs/database.yaml` and fill in your local PostgreSQL connection. Use the PostgreSQL databases, Qdrant collections, and index versions that match the workload. The four JSON templates in [configs/main](configs/main) specify the expected database and collection names, service URLs, and index locations. To change these for your deployment, copy the templates into a local directory and pass it via `--config-dir`.

| Workload | Published train / dev / test | External assets |
| --- | ---: | --- |
| arXiv | 565 / 185 / 200 | MiniLM/384 cosine corpus, PostgreSQL and Qdrant, FAISS index and ordinal-to-article ID receipts, and SQLite metadata under `assets/arxiv/` |
| Spider | 180 / 60 / 60 | MiniLM/384 cosine corpus, PostgreSQL databases, sharded Qdrant collections, FAISS indexes, the collection route map, and semantic metadata/catalog files under `assets/spider/` and `assets/index-services/` |
| Wikipedia | 176 / 60 / 60 | CLIP/512 cosine corpus, PostgreSQL and Qdrant, and `assets/wiki/completed.json` with the matching FAISS and row-ID paths |
| YFCC | 80 / 60 / 100 | YFCC1920 raw L2 vectors and scalar files, PostgreSQL, Qdrant, and the sharded FAISS index with its manifest under `assets/yfcc/` |

See [the workload manifest](data/published-workloads.json) for the exact query and split files. Corpora, model weights, databases, and prebuilt indexes are separate assets. Backend paths in the JSON templates are relative to `--assets-dir` where marked `@ASSETS@`. External index manifests and routing maps must also refer to the correct files and services on the new host.

The templates use MiniLM endpoints at ports 8365 (arXiv) and 8363 (Spider), and the CLIP endpoint at port 8361 (Wikipedia). The repository includes the corresponding encoder servers. With the matching local weights and `torch`, `sentence-transformers`, and `transformers` installed, start each command in its own terminal:

```bash
python services/minilm_server.py --model-path /path/to/minilm --device cpu --port 8365
python services/minilm_server.py --model-path /path/to/minilm --device cpu --port 8363
python services/wiki_embedding/clip_server.py --model-path /path/to/clip --device cpu --port 8361
```

For existing arXiv index snapshots, `python scripts/materialize_arxiv_id_mapping.py --help` shows how to derive the article ID map. For YFCC, `python scripts/build_yfcc_indexes.py --help` exposes the FAISS and ACORN index builders. The default YFCC raw input filenames are `base.10M.1920d.u8bin`, `base.metadata.10M.spmat`, `equal.u2bin`, and `range.f4bin`.
