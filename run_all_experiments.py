from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SEEDS = [42, 123, 2026]
SPLITS = ["random", "cross_drug", "cross_disease"]
VARIANTS = [
    "baseline_pure_gnn",
    "single_tower",
    "sota_dual_tower",
    "ablation_no_gnn",
    "variant_path_loss",
]

LOG_DIR = ROOT_DIR / "outputs" / "final_ablation_logs_20260329"
RUN_DIR = ROOT_DIR / "outputs" / "final_ablation_runs_20260329"
EVAL_DIR = ROOT_DIR / "outputs" / "final_ablation_pairfixed_ho_20260329"

TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train_quad_split_ho_probe.py"
HO_EVAL_SCRIPT = ROOT_DIR / "scripts" / "eval_pair_fixed_ho.py"

FEATURE_DIR = ROOT_DIR / "outputs" / "pubmedbert_hybrid_features_clean"
TRIPLET_TEXT_PKL = ROOT_DIR / "triplet_text_embeddings.pkl"
DRUG_MORGAN_PKL = ROOT_DIR / "drug_morgan_fingerprints.pkl"
DRUG_TEXT_PKL = ROOT_DIR / "thick_drug_text_embeddings_sapbert.pkl"
DISEASE_TEXT_PKL = ROOT_DIR / "thick_disease_text_embeddings_sapbert.pkl"
NODES_CSV = ROOT_DIR / "data" / "PrimeKG" / "nodes.csv"
EDGES_CSV = ROOT_DIR / "data" / "PrimeKG" / "edges.csv"


def get_processed_path(split: str) -> Path:
    mapping = {
        "random": ROOT_DIR / "data" / "PrimeKG" / "processed" / "primekg_indication_mvp.pt",
        "cross_drug": ROOT_DIR / "data" / "PrimeKG" / "processed" / "primekg_indication_cross_drug.pt",
        "cross_disease": ROOT_DIR / "data" / "PrimeKG" / "processed" / "primekg_indication_cross_disease.pt",
    }
    return mapping[split]


def get_ot_csv(split: str) -> Path:
    mapping = {
        "random": ROOT_DIR / "outputs" / "ot_random_external_profile_pair_clean" / "novel_ood_triplets.csv",
        "cross_drug": ROOT_DIR / "outputs" / "ot_cross_drug_external_profile_pair_clean" / "novel_ood_triplets.csv",
        "cross_disease": ROOT_DIR / "outputs" / "ot_cross_disease_external_profile_pair_clean" / "novel_ood_triplets.csv",
    }
    return mapping[split]


def get_variant_flags(variant: str) -> list[str]:
    if variant == "baseline_pure_gnn":
        return [
            "--disable-disease-semantic",
            "--path-loss-weight",
            "0.0",
        ]
    if variant == "single_tower":
        return [
            "--disease-text-embeddings-path",
            str(DISEASE_TEXT_PKL),
            "--path-loss-weight",
            "0.0",
        ]
    if variant == "sota_dual_tower":
        return [
            "--drug-text-embeddings-path",
            str(DRUG_TEXT_PKL),
            "--disease-text-embeddings-path",
            str(DISEASE_TEXT_PKL),
            "--path-loss-weight",
            "0.0",
        ]
    if variant == "ablation_no_gnn":
        return [
            "--drug-text-embeddings-path",
            str(DRUG_TEXT_PKL),
            "--disease-text-embeddings-path",
            str(DISEASE_TEXT_PKL),
            "--ablate-gnn",
            "--path-loss-weight",
            "0.0",
        ]
    if variant == "variant_path_loss":
        return [
            "--drug-text-embeddings-path",
            str(DRUG_TEXT_PKL),
            "--disease-text-embeddings-path",
            str(DISEASE_TEXT_PKL),
            "--path-loss-weight",
            "0.1",
        ]
    raise ValueError(f"Unknown variant: {variant}")


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=ROOT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(process.returncode)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("============================================================", flush=True)
    print("Final ablation run started", flush=True)
    print(f"ROOT_DIR={ROOT_DIR}", flush=True)
    print(f"LOG_DIR={LOG_DIR}", flush=True)
    print(f"RUN_DIR={RUN_DIR}", flush=True)
    print(f"EVAL_DIR={EVAL_DIR}", flush=True)
    print("============================================================", flush=True)

    for split in SPLITS:
        processed_path = get_processed_path(split)
        ot_csv = get_ot_csv(split)

        for variant in VARIANTS:
            variant_flags = get_variant_flags(variant)

            for seed in SEEDS:
                run_stem = f"{split}_{variant}_seed{seed}"
                train_log = LOG_DIR / f"{run_stem}_train.log"
                ho_log = LOG_DIR / f"{run_stem}_ho_eval.log"
                output_json = RUN_DIR / f"{run_stem}.json"
                checkpoint_path = RUN_DIR / f"{run_stem}.pt"
                ho_json = EVAL_DIR / f"{run_stem}.json"

                print("", flush=True)
                print("------------------------------------------------------------", flush=True)
                print(f"[RUN] split={split} variant={variant} seed={seed}", flush=True)
                print("------------------------------------------------------------", flush=True)

                train_cmd = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--processed-path",
                    str(processed_path),
                    "--output-json",
                    str(output_json),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--nodes-csv",
                    str(NODES_CSV),
                    "--edges-csv",
                    str(EDGES_CSV),
                    "--feature-dir",
                    str(FEATURE_DIR),
                    "--triplet-text-embeddings-path",
                    str(TRIPLET_TEXT_PKL),
                    "--drug-morgan-fingerprints-path",
                    str(DRUG_MORGAN_PKL),
                    "--text-distill-alpha",
                    "0.2",
                    "--primary-loss-type",
                    "bce",
                    "--epochs",
                    "60",
                    "--batch-size",
                    "32",
                    "--hidden-channels",
                    "128",
                    "--out-dim",
                    "128",
                    "--scorer-hidden-dim",
                    "128",
                    "--lr",
                    "1e-3",
                    "--weight-decay",
                    "1e-5",
                    "--dropout",
                    "0.1",
                    "--initial-residual-alpha",
                    "0.2",
                    "--encoder-type",
                    "rgcn",
                    "--agg-type",
                    "attention",
                    "--graph-surgery-mode",
                    "direct_only",
                    "--use-early-external-fusion",
                    "--dropedge-p",
                    "0.15",
                    "--seed",
                    str(seed),
                    "--ot-novel-csv",
                    str(ot_csv),
                ] + variant_flags

                print("[TRAIN] " + " ".join(train_cmd), flush=True)
                if run_logged(train_cmd, train_log) != 0:
                    print(f"[ERROR] Training failed for split={split} variant={variant} seed={seed}", flush=True)
                    print(f"        See log: {train_log}", flush=True)
                    continue

                if not checkpoint_path.exists():
                    print(f"[ERROR] Checkpoint missing after training: {checkpoint_path}", flush=True)
                    continue

                ho_cmd = [
                    sys.executable,
                    str(HO_EVAL_SCRIPT),
                    "--checkpoint-path",
                    str(checkpoint_path),
                    "--processed-path",
                    str(processed_path),
                    "--nodes-csv",
                    str(NODES_CSV),
                    "--edges-csv",
                    str(EDGES_CSV),
                    "--feature-dir",
                    str(FEATURE_DIR),
                    "--ot-novel-csv",
                    str(ot_csv),
                    "--graph-surgery-mode",
                    "direct_only",
                    "--batch-size",
                    "512",
                    "--seed",
                    str(seed),
                    "--output-json",
                    str(ho_json),
                ]
                if variant == "ablation_no_gnn":
                    ho_cmd.append("--ablate-gnn")

                print("[HO-EVAL] " + " ".join(ho_cmd), flush=True)
                if run_logged(ho_cmd, ho_log) != 0:
                    print(f"[ERROR] Pair-fixed HO eval failed for split={split} variant={variant} seed={seed}", flush=True)
                    print(f"        See log: {ho_log}", flush=True)
                    continue

                print(f"[DONE] split={split} variant={variant} seed={seed}", flush=True)
                print(f"       train_log={train_log}", flush=True)
                print(f"       ho_log={ho_log}", flush=True)
                print(f"       ckpt={checkpoint_path}", flush=True)
                print(f"       json={output_json}", flush=True)
                print(f"       ho_json={ho_json}", flush=True)

    print("============================================================", flush=True)
    print("Final ablation run finished", flush=True)
    print(f"Logs:  {LOG_DIR}", flush=True)
    print(f"Runs:  {RUN_DIR}", flush=True)
    print(f"HO:    {EVAL_DIR}", flush=True)
    print("============================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
