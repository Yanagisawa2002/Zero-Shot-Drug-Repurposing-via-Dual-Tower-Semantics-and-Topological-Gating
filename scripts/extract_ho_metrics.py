from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs/asym_route_dropedge_multiseed_20260327"
SPLITS = ("random", "cross_drug", "cross_disease")
SEEDS = (42, 43, 44)


def normalize_key(key: str) -> str:
    return key.lower().replace("-", "_").replace("@", "_at_").replace(" ", "_")


def iter_nested_items(obj: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            next_prefix = prefix + (key_str,)
            yield next_prefix, value
            yield from iter_nested_items(value, next_prefix)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            next_prefix = prefix + (str(index),)
            yield next_prefix, value
            yield from iter_nested_items(value, next_prefix)


def path_tail_matches(path: Sequence[str], candidate_tail: Sequence[str]) -> bool:
    if len(path) < len(candidate_tail):
        return False
    normalized_path = [normalize_key(part) for part in path[-len(candidate_tail):]]
    normalized_tail = [normalize_key(part) for part in candidate_tail]
    return normalized_path == normalized_tail


def extract_metric(payload: Dict[str, Any], metric_name: str, metric_aliases: Sequence[str]) -> float:
    preferred_tails = [
        ("ho_eval", "metrics", metric_name),
        ("ho_eval", metric_name),
        ("metrics", metric_name),
    ]
    alias_tails = [("ho_eval", "metrics", alias) for alias in metric_aliases]
    alias_tails += [("ho_eval", alias) for alias in metric_aliases]
    alias_tails += [("metrics", alias) for alias in metric_aliases]

    candidates: List[Tuple[Tuple[str, ...], float]] = []
    for path, value in iter_nested_items(payload):
        if not isinstance(value, (int, float)):
            continue
        if any(path_tail_matches(path, tail) for tail in preferred_tails):
            return float(value)
        if any(path_tail_matches(path, tail) for tail in alias_tails):
            candidates.append((path, float(value)))

    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        candidate_paths = ", ".join(".".join(path) for path, _ in candidates)
        raise KeyError(f"Ambiguous HO metric '{metric_name}'. Candidates: {candidate_paths}")

    raise KeyError(f"Could not find HO metric '{metric_name}' in payload.")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(stdev(values))


def format_mean_std(mu: float, sigma: float) -> str:
    if math.isnan(mu):
        return "NaN"
    return f"{mu:.4f} +/- {sigma:.4f}"


def print_markdown_table(rows: Sequence[Dict[str, str]]) -> None:
    print("| Split | HO AUPRC | HO MRR | HO Hit@1 |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(f"| `{row['split']}` | {row['auprc']} | {row['mrr']} | {row['hit_at_1']} |")


def main() -> None:
    summary_rows: List[Dict[str, str]] = []

    for split in SPLITS:
        auprc_values: List[float] = []
        mrr_values: List[float] = []
        hit_values: List[float] = []

        for seed in SEEDS:
            result_path = RESULTS_DIR / f"{split}_seed{seed}.json"
            payload = load_json(result_path)

            auprc = extract_metric(payload, metric_name="auprc", metric_aliases=("ho_auprc",))
            mrr = extract_metric(payload, metric_name="mrr", metric_aliases=("ho_mrr",))
            hit_at_1 = extract_metric(
                payload,
                metric_name="hit_at_1",
                metric_aliases=("hit@1", "hit_1", "ho_hit_at_1", "ho_hit@1", "ho_hit_1"),
            )

            auprc_values.append(auprc)
            mrr_values.append(mrr)
            hit_values.append(hit_at_1)

        auprc_mean, auprc_std = summarize(auprc_values)
        mrr_mean, mrr_std = summarize(mrr_values)
        hit_mean, hit_std = summarize(hit_values)

        summary_rows.append(
            {
                "split": split,
                "auprc": format_mean_std(auprc_mean, auprc_std),
                "mrr": format_mean_std(mrr_mean, mrr_std),
                "hit_at_1": format_mean_std(hit_mean, hit_std),
            }
        )

    print_markdown_table(summary_rows)


if __name__ == "__main__":
    main()

