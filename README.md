# UniPlan

**UniPlan: Optimizing the Performance and Recall of Filtered Vector Searches via A Unified Framework**

UniPlan optimizes filtered vector search (FVS) queries, including queries with complex relational predicates and joins. It jointly chooses a filtering strategy, a relational execution plan, and vector index settings. Its objective is to minimize end-to-end latency subject to the query's recall requirement.

## How UniPlan works

1. **UniSpace configuration.** UniSpace represents filtering strategies, SQL scan and join choices, and vector index parameters in one conditional configuration space. Candidate generation considers the settings that apply to each strategy.
2. **Safe reinforcement learning.** UniPlan uses recall and latency estimates to screen candidates for offline trials. It uses execution feedback to train an inference model that ranks configurations for new queries.
3. **Synergy inference.** A lightweight tree-based decision model determines whether to use a direct execution path. Otherwise, the offline-trained inference model selects a joint configuration. Queries do not require online tuning trials.

## Paper results

The paper compares UniPlan with ten baselines on **arXiv, Spider, Wikipedia, and YFCC**. Under the reported evaluation settings, UniPlan achieves a **100% recall-requirement pass rate** and reduces end-to-end latency by **up to 52%** relative to the state-of-the-art baseline. The gains depend on the workload and recall target. When a low-cost execution path already performs well, configuration selection can offset part of the execution savings.

## Quick start

From the repository root, use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

vec2plan plan \
  --query-source examples/queries.synthetic.jsonl \
  --split-json examples/splits.synthetic.json \
  --output-dir outputs/smoke-plan
```

The smoke check reads the synthetic queries and prints a JSON summary of the planned configurations and trial counts. It runs without data services or trained models. The Python package and CLI retain the existing names `vec2plan_bandit` and `vec2plan`.

## Experiments

For live experiments, install the extras required by your configuration, for example:

```bash
python -m pip install -e '.[postgres,faiss,acorn,model,baselines]'
```

Prepare the queries, fixed splits, indexes, and model artifacts required by the selected dataset configuration. The live backends can require PostgreSQL with pgvector, a FAISS index, Qdrant, and an embedding service. See the [data and index layout](data/README.md), [configuration guide](configs/README.md), and [experiment workflow](src/vec2plan_bandit/experiments/README.md) for setup and stage inputs. Some optional baselines require additional services or credentials.

```bash
python run.py plan --dataset configs/data.json
python run.py preflight --dataset configs/data.json
```

Once the required assets and services are ready, `run.py` can run `prepare`, `collect`, `train`, `dev`, and `test` individually. The default configuration is a frozen arXiv experiment with a recall target of 0.8 and a predicted quality threshold of 0.95. It writes new runs to `outputs/arxiv-recall08-restored/`. For that experiment, `python run.py evaluate` reuses the prepared artifacts and frozen model to run the dev and test stages. The default command covers the arXiv protocol. The paper's four-workload comparison uses workload-specific setups.

See the [results guide](results/README.md) for the available experiment archives and plotting commands. Raw data, indexes, model weights, and local service settings must be prepared separately.

## Repository map

| Path | Contents |
| --- | --- |
| [`src/vec2plan_bandit/`](src/vec2plan_bandit/README.md) | Configuration selection, inference, execution backends, and benchmarks |
| [`configs/`](configs/README.md) | Dataset paths, experiment settings, and baseline definitions |
| [`examples/`](examples/README.md) | Synthetic inputs, workload entry points, and plotting scripts |
| [`tests/`](tests/README.md) | Tests for the experiment workflow and core components |
| [`results/`](results/README.md) | Archived metrics, figures, and instructions for redrawing them |

<details>
<summary>Local arXiv experiment history</summary>

The archived arXiv test contains 200 queries with five repeats per method across ten methods. Its original metrics and figures are in `outputs/arxiv-recall08-resumed/test/` and `outputs/arxiv-recall08-resumed/figures/`. The frozen model is in `outputs/arxiv-recall08/refit/frozen_model/`. The restore manifest and backup are in `outputs/rollback-to-recall08/`. A new run's timings can vary with backend load.

The local cosine migration uses `all-MiniLM-L6-v2` with 384-dimensional vectors. See `configs/data.wiki-minilm.json` and `configs/data.bird.json` for the Wiki and BIRD settings. Local migration checks and reruns are under `outputs/cosine-migration/`, `outputs/wiki-minilm-cosine/`, and `outputs/bird-cosine-minilm/`.

</details>
