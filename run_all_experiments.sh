#!/usr/bin/env bash
set -u -o pipefail

# Final ablation launcher for the current repo.
# Note: this project does not expose a generic `train.py`; the actual entrypoint is
# `scripts/train_quad_split_ho_probe.py`. Likewise, pair-fixed HO evaluation uses
# `scripts/eval_pair_fixed_ho.py`.
#
# Variant semantics used here:
# - baseline_pure_gnn: disable all SapBERT text towers; keep the rest of the stack unchanged.
# - single_tower: disease SapBERT only.
# - sota_dual_tower: current dual-tower mainline, no auxiliary path loss.
# - ablation_no_gnn: dual-tower + `--ablate-gnn`.
# - variant_path_loss: dual-tower + path-level margin ranking loss.
#
# If your codebase later adds flags such as `--disable_drug_text`, replace the current
# omission-based logic below with those explicit flags.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

SEEDS=(42 123 2026)
SPLITS=("random" "cross_drug" "cross_disease")
VARIANTS=("baseline_pure_gnn" "single_tower" "sota_dual_tower" "ablation_no_gnn" "variant_path_loss")

LOG_DIR="$ROOT_DIR/outputs/final_ablation_logs_20260329"
RUN_DIR="$ROOT_DIR/outputs/final_ablation_runs_20260329"
EVAL_DIR="$ROOT_DIR/outputs/final_ablation_pairfixed_ho_20260329"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$EVAL_DIR"

TRAIN_SCRIPT="$ROOT_DIR/scripts/train_quad_split_ho_probe.py"
HO_EVAL_SCRIPT="$ROOT_DIR/scripts/eval_pair_fixed_ho.py"

FEATURE_DIR="$ROOT_DIR/outputs/pubmedbert_hybrid_features_clean"
TRIPLET_TEXT_PKL="$ROOT_DIR/triplet_text_embeddings.pkl"
DRUG_MORGAN_PKL="$ROOT_DIR/drug_morgan_fingerprints.pkl"
DRUG_TEXT_PKL="$ROOT_DIR/thick_drug_text_embeddings_sapbert.pkl"
DISEASE_TEXT_PKL="$ROOT_DIR/thick_disease_text_embeddings_sapbert.pkl"
NODES_CSV="$ROOT_DIR/data/PrimeKG/nodes.csv"
EDGES_CSV="$ROOT_DIR/data/PrimeKG/edges.csv"

get_processed_path() {
  local split="$1"
  case "$split" in
    random) echo "$ROOT_DIR/data/PrimeKG/processed/primekg_indication_mvp.pt" ;;
    cross_drug) echo "$ROOT_DIR/data/PrimeKG/processed/primekg_indication_cross_drug.pt" ;;
    cross_disease) echo "$ROOT_DIR/data/PrimeKG/processed/primekg_indication_cross_disease.pt" ;;
    *) echo "" ; return 1 ;;
  esac
}

get_ot_csv() {
  local split="$1"
  case "$split" in
    random) echo "$ROOT_DIR/outputs/ot_random_external_profile_pair_clean/novel_ood_triplets.csv" ;;
    cross_drug) echo "$ROOT_DIR/outputs/ot_cross_drug_external_profile_pair_clean/novel_ood_triplets.csv" ;;
    cross_disease) echo "$ROOT_DIR/outputs/ot_cross_disease_external_profile_pair_clean/novel_ood_triplets.csv" ;;
    *) echo "" ; return 1 ;;
  esac
}

build_variant_flags() {
  local variant="$1"
  VARIANT_FLAGS=()
  case "$variant" in
    baseline_pure_gnn)
      VARIANT_FLAGS+=(
        --disable-disease-semantic
        --path-loss-weight 0.0
      )
      ;;
    single_tower)
      VARIANT_FLAGS+=(
        --disease-text-embeddings-path "$DISEASE_TEXT_PKL"
        --path-loss-weight 0.0
      )
      ;;
    sota_dual_tower)
      VARIANT_FLAGS+=(
        --drug-text-embeddings-path "$DRUG_TEXT_PKL"
        --disease-text-embeddings-path "$DISEASE_TEXT_PKL"
        --path-loss-weight 0.0
      )
      ;;
    ablation_no_gnn)
      VARIANT_FLAGS+=(
        --drug-text-embeddings-path "$DRUG_TEXT_PKL"
        --disease-text-embeddings-path "$DISEASE_TEXT_PKL"
        --ablate-gnn
        --path-loss-weight 0.0
      )
      ;;
    variant_path_loss)
      VARIANT_FLAGS+=(
        --drug-text-embeddings-path "$DRUG_TEXT_PKL"
        --disease-text-embeddings-path "$DISEASE_TEXT_PKL"
        --path-loss-weight 0.1
      )
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 1
      ;;
  esac
}

echo "============================================================"
echo "Final ablation run started"
echo "ROOT_DIR=$ROOT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "EVAL_DIR=$EVAL_DIR"
echo "============================================================"

for split in "${SPLITS[@]}"; do
  processed_path="$(get_processed_path "$split")" || {
    echo "[ERROR] Could not resolve processed path for split=$split"
    continue
  }
  ot_csv="$(get_ot_csv "$split")" || {
    echo "[ERROR] Could not resolve OT CSV for split=$split"
    continue
  }

  for variant in "${VARIANTS[@]}"; do
    if ! build_variant_flags "$variant"; then
      echo "[ERROR] Failed to build flags for variant=$variant"
      continue
    fi

    for seed in "${SEEDS[@]}"; do
      echo
      echo "------------------------------------------------------------"
      echo "[RUN] split=$split variant=$variant seed=$seed"
      echo "------------------------------------------------------------"

      run_stem="${split}_${variant}_seed${seed}"
      train_log="$LOG_DIR/${run_stem}_train.log"
      ho_log="$LOG_DIR/${run_stem}_ho_eval.log"
      output_json="$RUN_DIR/${run_stem}.json"
      checkpoint_path="$RUN_DIR/${run_stem}.pt"
      ho_json="$EVAL_DIR/${run_stem}.json"

      train_cmd=(
        python "$TRAIN_SCRIPT"
        --processed-path "$processed_path"
        --output-json "$output_json"
        --checkpoint-path "$checkpoint_path"
        --nodes-csv "$NODES_CSV"
        --edges-csv "$EDGES_CSV"
        --feature-dir "$FEATURE_DIR"
        --triplet-text-embeddings-path "$TRIPLET_TEXT_PKL"
        --drug-morgan-fingerprints-path "$DRUG_MORGAN_PKL"
        --text-distill-alpha 0.2
        --primary-loss-type bce
        --epochs 60
        --batch-size 32
        --hidden-channels 128
        --out-dim 128
        --scorer-hidden-dim 128
        --lr 1e-3
        --weight-decay 1e-5
        --dropout 0.1
        --initial-residual-alpha 0.2
        --encoder-type rgcn
        --agg-type attention
        --graph-surgery-mode direct_only
        --use-early-external-fusion
        --dropedge-p 0.15
        --seed "$seed"
        --ot-novel-csv "$ot_csv"
      )
      train_cmd+=("${VARIANT_FLAGS[@]}")

      echo "[TRAIN] ${train_cmd[*]}"
      if ! "${train_cmd[@]}" >"$train_log" 2>&1; then
        echo "[ERROR] Training failed for split=$split variant=$variant seed=$seed"
        echo "        See log: $train_log"
        continue
      fi

      if [[ ! -f "$checkpoint_path" ]]; then
        echo "[ERROR] Checkpoint missing after training: $checkpoint_path"
        continue
      fi

      ho_cmd=(
        python "$HO_EVAL_SCRIPT"
        --checkpoint-path "$checkpoint_path"
        --processed-path "$processed_path"
        --nodes-csv "$NODES_CSV"
        --edges-csv "$EDGES_CSV"
        --feature-dir "$FEATURE_DIR"
        --ot-novel-csv "$ot_csv"
        --graph-surgery-mode direct_only
        --batch-size 512
        --seed "$seed"
        --output-json "$ho_json"
      )
      if [[ "$variant" == "ablation_no_gnn" ]]; then
        ho_cmd+=(--ablate-gnn)
      fi

      echo "[HO-EVAL] ${ho_cmd[*]}"
      if ! "${ho_cmd[@]}" >"$ho_log" 2>&1; then
        echo "[ERROR] Pair-fixed HO eval failed for split=$split variant=$variant seed=$seed"
        echo "        See log: $ho_log"
        continue
      fi

      echo "[DONE] split=$split variant=$variant seed=$seed"
      echo "       train_log=$train_log"
      echo "       ho_log=$ho_log"
      echo "       ckpt=$checkpoint_path"
      echo "       json=$output_json"
      echo "       ho_json=$ho_json"
    done
  done
done

echo "============================================================"
echo "Final ablation run finished"
echo "Logs:  $LOG_DIR"
echo "Runs:  $RUN_DIR"
echo "HO:    $EVAL_DIR"
echo "============================================================"
