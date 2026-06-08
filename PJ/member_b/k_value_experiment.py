import argparse
import html
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score


CUSTOM_STOP_WORDS = {
    "stackoverflow",
    "question",
    "answer",
    "thanks",
    "please",
    "help",
    "error",
    "problem",
    "code",
    "using",
    "use",
    "want",
    "need",
}


def clean_text(value: str) -> str:
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def read_json_records(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        first = ""
        while not first:
            first = f.read(1)
        f.seek(0)
        if first == "[":
            return pd.DataFrame(json.load(f))
        rows = []
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return read_json_records(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def build_text_column(df: pd.DataFrame) -> pd.Series:
    lower_cols = {c.lower(): c for c in df.columns}

    if "text" in lower_cols:
        return df[lower_cols["text"]].fillna("")
    if "question" in lower_cols:
        return df[lower_cols["question"]].fillna("")
    if "title" in lower_cols and "body" in lower_cols:
        return (
            df[lower_cols["title"]].fillna("").astype(str)
            + " "
            + df[lower_cols["body"]].fillna("").astype(str)
        )
    if "body" in lower_cols:
        return df[lower_cols["body"]].fillna("")
    if "title" in lower_cols:
        return df[lower_cols["title"]].fillna("")

    raise ValueError(
        "No valid text column found. Expected one of: text, question, title, body."
    )


def cluster_size_summary(labels: np.ndarray) -> dict:
    counts = np.bincount(labels)
    return {
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "median_cluster_size": float(np.median(counts)),
        "small_cluster_count_lt_10": int((counts < 10).sum()),
    }


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    df = read_input(Path(args.input))
    print(f"Loaded {len(df)} records from {args.input}")

    if args.sample_ratio < 1.0:
        df = df.sample(frac=args.sample_ratio, random_state=args.random_state)
        print(f"Sampled {len(df)} records (ratio={args.sample_ratio})")

    raw_texts = build_text_column(df)
    texts = raw_texts.map(clean_text)
    texts = texts[texts.str.len() >= args.min_text_length]
    print(f"Usable documents after preprocessing: {len(texts)}")

    if len(texts) < args.k_max:
        raise ValueError(
            f"Too few usable documents ({len(texts)}) for k_max={args.k_max}."
        )

    stop_words = sorted(set(ENGLISH_STOP_WORDS) | CUSTOM_STOP_WORDS)
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=stop_words,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9+#.\-]{1,}\b",
        ngram_range=(1, args.max_ngram),
        sublinear_tf=True,
    )
    x = vectorizer.fit_transform(texts)

    results = []
    for k in range(args.k_min, args.k_max + 1):
        started = time.perf_counter()
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=args.n_init,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        labels = model.fit_predict(x)
        elapsed = time.perf_counter() - started

        sample_size = min(args.silhouette_sample, x.shape[0])
        silhouette = silhouette_score(
            x,
            labels,
            metric=args.silhouette_metric,
            sample_size=sample_size,
            random_state=args.random_state,
        )
        size_summary = cluster_size_summary(labels)

        results.append(
            {
                "k": k,
                "silhouette_score": round(float(silhouette), 6),
                "inertia": round(float(model.inertia_), 3),
                "runtime_seconds": round(float(elapsed), 3),
                **size_summary,
            }
        )
        print(f"K={k}: silhouette={silhouette:.4f}, inertia={model.inertia_:.2f}")

    return pd.DataFrame(results)


def save_plot(results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(results["k"], results["silhouette_score"], marker="o")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("K Value Selection Metrics")
    axes[0].grid(alpha=0.25)

    axes[1].plot(results["k"], results["inertia"], marker="o", color="#d65f00")
    axes[1].set_ylabel("SSE / Inertia")
    axes[1].grid(alpha=0.25)

    axes[2].plot(results["k"], results["runtime_seconds"], marker="o", color="#2271b2")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Runtime (s)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "k_value_metrics.png", dpi=180)
    plt.close(fig)


def choose_best_k(results: pd.DataFrame, baseline_k: int, tolerance: float) -> tuple[int, str]:
    best_row = results.sort_values(
        ["silhouette_score", "small_cluster_count_lt_10"],
        ascending=[False, True],
    ).iloc[0]
    best_k = int(best_row["k"])

    top_score = float(best_row["silhouette_score"])
    close = results[results["silhouette_score"] >= top_score - tolerance]
    baseline_rows = close[close["k"] == baseline_k]

    if not baseline_rows.empty:
        baseline_score = float(baseline_rows.iloc[0]["silhouette_score"])
        reason = (
            f"K={best_k} has the highest silhouette score ({top_score:.4f}), "
            f"while baseline K={baseline_k} is within {tolerance:.4f} "
            f"({baseline_score:.4f}) and keeps experiments comparable."
        )
        return baseline_k, reason

    simpler_k = int(close.sort_values("k").iloc[0]["k"])
    if simpler_k != best_k:
        reason = (
            f"K={best_k} has the highest silhouette score ({top_score:.4f}), "
            f"but K={simpler_k} is within {tolerance:.4f} and gives a simpler clustering structure."
        )
        return simpler_k, reason

    reason = (
        f"K={best_k} has the best overall silhouette score ({top_score:.4f}) "
        "and is selected as the recommended cluster number."
    )
    return best_k, reason


def main() -> None:
    parser = argparse.ArgumentParser(description="K value selection experiment.")
    parser.add_argument("--input", required=True, help="Input CSV/JSONL/Parquet file.")
    parser.add_argument("--output", default="member_b/outputs", help="Output folder.")
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=60)
    parser.add_argument("--sample-ratio", type=float, default=0.1, help="Sampling ratio (default: 0.1, same as baseline cluster run)")
    parser.add_argument("--max-features", type=int, default=5000, help="Match baseline default")
    parser.add_argument("--min-df", type=int, default=5, help="Match baseline default")
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-ngram", type=int, default=1)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=30, help="Match baseline default")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--baseline-k", type=int, default=50, help="Prefer this K when its score is close to the best score")
    parser.add_argument("--selection-tolerance", type=float, default=0.005, help="Score gap treated as close enough for simpler/comparable K selection")
    parser.add_argument("--silhouette-sample", type=int, default=5000)
    parser.add_argument("--silhouette-metric", default="cosine", choices=["cosine", "euclidean"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(args)
    results.to_csv(output_dir / "k_value_results.csv", index=False)
    save_plot(results, output_dir)

    best_k, reason = choose_best_k(results, args.baseline_k, args.selection_tolerance)
    with (output_dir / "best_k_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Recommended K: {best_k}\n")
        f.write(reason + "\n")

    print(f"\nRecommended K: {best_k}")
    print(reason)


if __name__ == "__main__":
    main()
