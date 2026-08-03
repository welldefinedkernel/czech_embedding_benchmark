"""Create a synthetic golden dataset from the Czech Text Document Corpus (CTDC)."""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

from backends.azure_llm import AzureLLM

SOURCES_DIR = Path("data/czech_text_document_corpus_v20/sources")
OUTPUT_DIR = Path("data/czech_text_document_corpus_v20/synthetic")

MIN_WORDS = 100
OPENING_TOKENS = 8

STYLING_CONFIG = StylingConfig(
    scenario=(
        "A Czech speaker asking a question about a specific event, fact or figure "
        "reported in one article of the ČTK news agency archive."
    ),
    task=(
        "Retrieve the single most relevant Czech news article from the archive and "
        "answer the user's question from it."
    ),
    input_format=(
        "A single self-contained question in Czech, between 6 and 20 words, ending "
        "with a question mark. It must ask about exactly one fact and be answerable in "
        "one or two sentences. It must never chain several sub-questions together with "
        "commas, colons, or connectives such as 'a kdy', 'a jak', 'a kolik'. It must "
        "paraphrase the article in the user's own words instead of quoting the headline "
        "or any sentence verbatim. It must never contain instructions such as 'shrň', "
        "'ověř', 'vytvoř tabulku' or 'představ si', must never mention 'text', "
        "'kontext', 'článek' or 'dokument', and must not use markdown. It must name "
        "enough entities and details that only this one article answers it."
    ),
    expected_output_format=(
        "A short factual answer in Czech, at most two sentences, stating only the fact "
        "the question asks for, with no extra commentary. Plain text only, no markdown "
        "formatting, no bullet lists and no tables."
    ),
)


def theme_of(path: Path) -> str:
    """Return the primary CTDC category encoded in the file name."""
    return path.stem.split("_")[1]


def _opening_key(text: str) -> str:
    """Build a digit-insensitive fingerprint of a document's opening words."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    words = [word for word in words if not any(char.isdigit() for char in word)]
    return " ".join(words[:OPENING_TOKENS])


def load_pool(sources_dir: Path) -> dict[Path, str]:
    """Read documents, dropping stubs and repeated ČTK boilerplate templates."""
    pool: dict[Path, str] = {}
    seen_openings: set[str] = set()

    for path in sorted(sources_dir.glob("*.txt")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        if len(text.split()) < MIN_WORDS:
            continue
        opening = _opening_key(text)
        if opening in seen_openings:
            continue
        seen_openings.add(opening)
        pool[path] = text

    if not pool:
        raise FileNotFoundError(f"No usable .txt documents found in {sources_dir}")
    return pool


def sample_documents(
    sources_dir: Path, count: int, seed: int
) -> list[tuple[str, str, str]]:
    """Sample documents while keeping the corpus-wide theme proportions."""
    pool = load_pool(sources_dir)

    by_theme: dict[str, list[Path]] = defaultdict(list)
    for path in pool:
        by_theme[theme_of(path)].append(path)

    total = len(pool)
    count = min(count, total)

    # Largest-remainder apportionment so the per-theme quotas sum exactly to `count`.
    exact = {theme: len(paths) * count / total for theme, paths in by_theme.items()}
    quotas = {theme: int(share) for theme, share in exact.items()}
    remaining = count - sum(quotas.values())
    for theme in sorted(exact, key=lambda t: exact[t] - quotas[t], reverse=True):
        if remaining <= 0:
            break
        if quotas[theme] < len(by_theme[theme]):
            quotas[theme] += 1
            remaining -= 1

    rng = random.Random(seed)
    documents: list[tuple[str, str, str]] = []
    for theme in sorted(quotas):
        for path in rng.sample(by_theme[theme], quotas[theme]):
            documents.append((path.stem, theme, pool[path]))

    rng.shuffle(documents)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", type=int, required=True, help="Number of CTDC documents to use."
    )
    parser.add_argument(
        "--goldens-per-document",
        type=int,
        required=True,
        help="Maximum number of goldens generated per document.",
    )
    parser.add_argument(
        "--no-expected-output",
        action="store_true",
        help="Skip answer generation, e.g. for cheap test runs.",
    )
    parser.add_argument("--sources-dir", type=Path, default=SOURCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--name", default="ctdc_synthetic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrent", type=int, default=100)
    args = parser.parse_args()

    if args.documents < 1 or args.goldens_per_document < 1:
        parser.error("--documents and --goldens-per-document must be >= 1")

    documents = sample_documents(args.sources_dir, args.documents, args.seed)
    themes = {theme for _, theme, _ in documents}
    print(f"Sampled {len(documents)} documents across {len(themes)} themes.")

    synthesizer = Synthesizer(
        model=AzureLLM(),
        max_concurrent=args.max_concurrent,
        styling_config=STYLING_CONFIG,
    )
    synthesizer.generate_goldens_from_contexts(
        contexts=[[text] for _, _, text in documents],
        include_expected_output=not args.no_expected_output,
        max_goldens_per_context=args.goldens_per_document,
        source_files=[document_id for document_id, _, _ in documents],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer.save_as(
        file_type="json",
        directory=str(args.output_dir),
        file_name=args.name,
    )


if __name__ == "__main__":
    main()
