# Evaluation and comparison of multilingual embedding models on Czech datasets

## Setup & Usage

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run run.py --config configs/retrieval_evaluation.toml
```

See [configs/retrieval_evaluation.toml.example](configs/retrieval_evaluation.toml.example) for config options.

## Results

Full leaderboards are in **[results/README.md](results/README.md)**, covering:

1. **MS MARCO — Czech** — Czech queries against Czech passages
2. **MS MARCO — English** — English queries against English passages
3. **CTDC synthetic — Czech** — synthetic Czech questions against native Czech news articles
4. **MTEB (Multilingual, v2)** — official retrieval subset
