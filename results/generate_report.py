"""Render this results folder into README.md.

GitHub shows that file when the results folder is opened.

Usage:
    python3 results/generate_report.py
"""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = RESULTS_ROOT / "mteb"
OUTPUT_FILE = RESULTS_ROOT / "README.md"

MSMARCO_DIR = RESULTS_DIR / "msmarco"
CTDC_DIR = RESULTS_DIR / "ctdc_synthetic"
CTDC_RERANK_DIR = RESULTS_DIR / "ctdc_synthetic_rerank"
MTEB_DIR = RESULTS_DIR / "mteb_multilingual_retrieval"

MSMARCO_CZ_TASK = "msmarco_validation_cz-cz_retrieval"
MSMARCO_EN_TASK = "msmarco_validation_en-en_retrieval"
CTDC_TASK = "ctdc_synthetic_retrieval"

RETRIEVAL_METRICS = (
    ("mrr_at_10", "MRR@10"),
    ("ndcg_at_10", "nDCG@10"),
    ("map_at_10", "MAP@10"),
    ("recall_at_1", "Recall@1"),
    ("recall_at_10", "Recall@10"),
    ("recall_at_20", "Recall@20"),
    ("recall_at_100", "Recall@100"),
)


def load_model_results(directory: Path) -> list[dict]:
    """Load every metrics.json in a results directory."""
    results = []
    for metrics_file in sorted(directory.glob("*/metrics.json")):
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metrics["_dir"] = metrics_file.parent
        results.append(metrics)
    return results


def load_rerank_results(directory: Path) -> list[dict]:
    """Load reranked runs, which nest one level deeper as <first stage>/<reranker>."""
    results = []
    for metrics_file in sorted(directory.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        metrics["_dir"] = metrics_file.parent
        # `model_name` here is the reranker; the retriever is the parent folder.
        metrics["_first_stage"] = metrics_file.parent.parent.name.replace("__", "/")
        results.append(metrics)
    return results


def score_entries(task_result: dict) -> list[dict]:
    """Flatten a task result into its per-split, per-subset score entries."""
    return [entry for split in task_result["scores"].values() for entry in split]


def task_score(task_result: dict) -> float:
    """Mean main_score over all splits and subsets of a task."""
    entries = score_entries(task_result)
    return sum(entry["main_score"] for entry in entries) / len(entries)


def find_task(metrics: dict, task_name: str) -> dict | None:
    for task_result in metrics["task_results"]:
        if task_result["task_name"] == task_name:
            return task_result
    return None


def format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def model_link(metrics: dict) -> str:
    """Model name linking to its result folder, relative to README.md."""
    relative = metrics["_dir"].relative_to(RESULTS_ROOT)
    return f"[`{metrics['model_name']}`]({relative}/)"


def render_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render_retrieval_section(
    all_metrics: list[dict], task_name: str, title: str, description: str
) -> str:
    rows = []
    for metrics in all_metrics:
        task_result = find_task(metrics, task_name)
        if task_result is None:
            continue
        entry = score_entries(task_result)[0]
        rows.append(
            {
                "model": model_link(metrics),
                "scores": [entry.get(key) for key, _ in RETRIEVAL_METRICS],
                "date": task_result["date"][:10],
            }
        )

    rows.sort(key=lambda row: row["scores"][0] or 0.0, reverse=True)

    header = ["#", "Model"] + [label for _, label in RETRIEVAL_METRICS] + ["Date"]
    table_rows = [
        [str(rank), row["model"]]
        + [format_score(score) for score in row["scores"]]
        + [row["date"]]
        for rank, row in enumerate(rows, start=1)
    ]

    return f"## {title}\n\n{description}\n\n" + render_table(header, table_rows)


def render_ctdc_rerank_section(
    rerank_metrics: list[dict], first_stage_metrics: list[dict]
) -> str:
    first_stage = {}
    for metrics in first_stage_metrics:
        task_result = find_task(metrics, CTDC_TASK)
        if task_result is not None:
            first_stage[metrics["model_name"]] = score_entries(task_result)[0]

    by_reranker: dict[str, list[dict]] = defaultdict(list)
    for metrics in rerank_metrics:
        if find_task(metrics, CTDC_TASK) is not None:
            by_reranker[metrics["model_name"]].append(metrics)

    if not by_reranker:
        return ""

    header = (
        ["#", "Retriever"]
        + [label for _, label in RETRIEVAL_METRICS]
        + ["ΔMRR@10", "Date"]
    )

    blocks = []
    for reranker, runs in by_reranker.items():
        # A partial sweep covers whichever retrievers finished first, so its mean
        # is not comparable with a complete one; leave it out until it finishes.
        if len(runs) != len(first_stage):
            continue
        rows = []
        for metrics in runs:
            task_result = find_task(metrics, CTDC_TASK)
            entry = score_entries(task_result)[0]
            base = first_stage.get(metrics["_first_stage"], {}).get("mrr_at_10")
            rows.append(
                {
                    "model": f"[`{metrics['_first_stage']}`]"
                    f"({metrics['_dir'].relative_to(RESULTS_ROOT)}/)",
                    "scores": [entry.get(key) for key, _ in RETRIEVAL_METRICS],
                    "delta": None if base is None else entry["mrr_at_10"] - base,
                    "date": task_result["date"][:10],
                }
            )
        rows.sort(key=lambda row: row["scores"][0] or 0.0, reverse=True)
        table_rows = [
            [str(rank), row["model"]]
            + [format_score(score) for score in row["scores"]]
            + ["—" if row["delta"] is None else f"{row['delta']:+.4f}", row["date"]]
            for rank, row in enumerate(rows, start=1)
        ]
        mean_mrr = sum(row["scores"][0] for row in rows) / len(rows)
        blocks.append(
            (
                mean_mrr,
                f"### [`{reranker}`](https://huggingface.co/{reranker})\n\n"
                + render_table(header, table_rows),
            )
        )

    if not blocks:
        return ""

    blocks.sort(key=lambda block: block[0], reverse=True)

    description = (
        "Each retriever's top 100 documents rescored by a cross-encoder. ΔMRR@10 is measured "
        "against section 3. Recall@100 is unchanged."
    )

    return f"## 4. CTDC synthetic — after reranking\n\n{description}\n\n" + "\n\n".join(
        block for _, block in blocks
    )


def render_mteb_section(all_metrics: list[dict]) -> str:
    models = []
    for metrics in all_metrics:
        scores = {
            task_result["task_name"]: task_score(task_result)
            for task_result in metrics["task_results"]
        }
        models.append(
            {
                "model": metrics["model_name"],
                "link": model_link(metrics),
                "scores": scores,
                "mean": sum(scores.values()) / len(scores),
            }
        )

    models.sort(key=lambda row: row["mean"], reverse=True)

    summary = render_table(
        ["#", "Model", "Mean(Task)", "Tasks"],
        [
            [
                str(rank),
                row["link"],
                format_score(row["mean"]),
                str(len(row["scores"])),
            ]
            for rank, row in enumerate(models, start=1)
        ],
    )

    task_names = sorted({name for row in models for name in row["scores"]})
    per_task = render_table(
        ["Task"] + [f"`{row['model']}`" for row in models],
        [
            [name] + [format_score(row["scores"].get(name)) for row in models]
            for name in task_names
        ],
    )

    return (
        "## 5. MTEB (Multilingual, v2) — retrieval subset\n\n"
        "Official MTEB retrieval tasks, scored with each task's own main metric "
        "and averaged over all splits and subsets.\n\n"
        f"{summary}\n\n### Per-task scores\n\n{per_task}"
    )


def main() -> None:
    msmarco_metrics = load_model_results(MSMARCO_DIR)
    ctdc_metrics = load_model_results(CTDC_DIR)
    ctdc_rerank_metrics = load_rerank_results(CTDC_RERANK_DIR)
    mteb_metrics = load_model_results(MTEB_DIR)

    sections = [
        "# Benchmark Results",
        render_retrieval_section(
            msmarco_metrics,
            MSMARCO_CZ_TASK,
            "1. MS MARCO — Czech",
            "Czech queries against Czech passages (machine-translated MS MARCO "
            "validation split).",
        ),
        render_retrieval_section(
            msmarco_metrics,
            MSMARCO_EN_TASK,
            "2. MS MARCO — English",
            "Original English queries against English passages (MS MARCO "
            "validation split).",
        ),
        render_retrieval_section(
            ctdc_metrics,
            CTDC_TASK,
            "3. CTDC synthetic — Czech",
            "Synthetic Czech questions against the Czech Text Document Corpus "
            "(14,887 queries over a 14,690-document corpus of native Czech news "
            "articles).",
        ),
        render_ctdc_rerank_section(ctdc_rerank_metrics, ctdc_metrics),
        render_mteb_section(mteb_metrics),
    ]

    readme = "\n\n".join(section for section in sections if section) + "\n"
    OUTPUT_FILE.write_text(readme, encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE.relative_to(RESULTS_ROOT.parent)}")


if __name__ == "__main__":
    main()
