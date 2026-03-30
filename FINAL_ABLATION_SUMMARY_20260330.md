# Final Ablation Summary (2026-03-30)

This report aggregates the completed `45` experiment groups from:

- ``outputs/final_ablation_runs_20260329``
- ``outputs/final_ablation_pairfixed_ho_20260329``

Settings:

- Seeds: `42, 123, 2026`
- Splits: `random, cross_drug, cross_disease`
- Variants:
  - `baseline_pure_gnn`
  - `single_tower`
  - `sota_dual_tower`
  - `ablation_no_gnn`
  - `variant_path_loss`

Metric conventions:

- `Pair AUROC`: internal pairwise test AUROC under the split's own protocol
- `OT AUROC`: Open Targets external AUROC under the same split
- `HO AUPRC`: original HO full-candidate mechanism evaluation AUPRC
- `Pair-Fixed HO`: mechanism-only evaluation keeping the same `(drug, disease)` and comparing only `positive / neg_gene / neg_pathway`

## Random

| Variant | Pair AUROC | OT AUROC | HO AUPRC | Pair-Fixed HO AUPRC | Pair-Fixed Hit@1 | Pair-Fixed MRR |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pure_gnn | 0.9786 ¡À 0.0060 | 0.8886 ¡À 0.0120 | 0.3409 ¡À 0.0070 | 0.3752 ¡À 0.0101 | 0.3435 ¡À 0.0491 | 0.6457 ¡À 0.0311 |
| single_tower | 0.9860 ¡À 0.0030 | 0.8739 ¡À 0.0080 | 0.3755 ¡À 0.0032 | 0.3912 ¡À 0.0062 | 0.3523 ¡À 0.0185 | 0.6499 ¡À 0.0142 |
| sota_dual_tower | 0.9873 ¡À 0.0018 | 0.8500 ¡À 0.0109 | 0.3323 ¡À 0.0054 | 0.3353 ¡À 0.0060 | 0.1796 ¡À 0.0788 | 0.5260 ¡À 0.0599 |
| ablation_no_gnn | 0.9818 ¡À 0.0030 | 0.8008 ¡À 0.0096 | 0.3290 ¡À 0.0003 | 0.3333 ¡À 0.0000 | 0.3773 ¡À 0.0103 | 0.6398 ¡À 0.0066 |
| variant_path_loss | 0.9895 ¡À 0.0015 | 0.8591 ¡À 0.0109 | 0.3381 ¡À 0.0055 | 0.3416 ¡À 0.0059 | 0.2925 ¡À 0.0250 | 0.6097 ¡À 0.0144 |

## Cross-Drug

| Variant | Pair AUROC | OT AUROC | HO AUPRC | Pair-Fixed HO AUPRC | Pair-Fixed Hit@1 | Pair-Fixed MRR |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pure_gnn | 0.8368 ¡À 0.0244 | 0.8335 ¡À 0.0064 | 0.2934 ¡À 0.0163 | 0.3584 ¡À 0.0037 | 0.2231 ¡À 0.0018 | 0.5857 ¡À 0.0023 |
| single_tower | 0.8436 ¡À 0.0125 | 0.8256 ¡À 0.0080 | 0.3223 ¡À 0.0060 | 0.3647 ¡À 0.0127 | 0.2919 ¡À 0.0391 | 0.6228 ¡À 0.0217 |
| sota_dual_tower | 0.8854 ¡À 0.0202 | 0.8122 ¡À 0.0161 | 0.3194 ¡À 0.0072 | 0.3378 ¡À 0.0056 | 0.2689 ¡À 0.0661 | 0.6020 ¡À 0.0495 |
| ablation_no_gnn | 0.8336 ¡À 0.0130 | 0.7228 ¡À 0.0060 | 0.3077 ¡À 0.0028 | 0.3333 ¡À 0.0000 | 0.3723 ¡À 0.0056 | 0.6374 ¡À 0.0031 |
| variant_path_loss | 0.8846 ¡À 0.0196 | 0.8129 ¡À 0.0144 | 0.3193 ¡À 0.0027 | 0.3413 ¡À 0.0045 | 0.2280 ¡À 0.0217 | 0.5833 ¡À 0.0127 |

## Cross-Disease

| Variant | Pair AUROC | OT AUROC | HO AUPRC | Pair-Fixed HO AUPRC | Pair-Fixed Hit@1 | Pair-Fixed MRR |
|---|---:|---:|---:|---:|---:|---:|
| baseline_pure_gnn | 0.6423 ¡À 0.0211 | 0.7482 ¡À 0.0199 | 0.2409 ¡À 0.0167 | 0.3539 ¡À 0.0164 | 0.2672 ¡À 0.0397 | 0.5956 ¡À 0.0281 |
| single_tower | 0.7613 ¡À 0.0523 | 0.7355 ¡À 0.0304 | 0.3236 ¡À 0.0165 | 0.3595 ¡À 0.0119 | 0.2639 ¡À 0.0353 | 0.5970 ¡À 0.0222 |
| sota_dual_tower | 0.8014 ¡À 0.0134 | 0.6956 ¡À 0.0240 | 0.3078 ¡À 0.0039 | 0.3320 ¡À 0.0043 | 0.2149 ¡À 0.0202 | 0.5492 ¡À 0.0045 |
| ablation_no_gnn | 0.8828 ¡À 0.0154 | 0.7236 ¡À 0.0063 | 0.3197 ¡À 0.0019 | 0.3333 ¡À 0.0000 | 0.3650 ¡À 0.0085 | 0.6321 ¡À 0.0057 |
| variant_path_loss | 0.8224 ¡À 0.0074 | 0.7161 ¡À 0.0089 | 0.3193 ¡À 0.0091 | 0.3380 ¡À 0.0078 | 0.2531 ¡À 0.0496 | 0.5847 ¡À 0.0341 |

## Main Findings

1. The dual-tower model is not universally dominant.
   - It is strongest on internal pairwise AUROC for `random` and `cross_drug`.
   - It is not strongest on `cross_disease`; there, `ablation_no_gnn` gives the highest pairwise and OT AUROC.

2. Disease-side semantics contribute much more than drug-side semantics in the hardest OOD setting.
   - `cross_disease` jumps from `0.6423` to `0.7613` in Pair AUROC when moving from `baseline_pure_gnn` to `single_tower`.
   - This indicates disease SapBERT is the main driver of disease-side zero-shot generalization.

3. Adding drug-side SapBERT helps internal pairwise ranking but hurts external OT in all three splits.
   - `single_tower -> sota_dual_tower`:
     - `random` OT AUROC: `0.8739 -> 0.8500`
     - `cross_drug` OT AUROC: `0.8256 -> 0.8122`
     - `cross_disease` OT AUROC: `0.7355 -> 0.6956`
   - The drug text tower improves symmetry, but currently shifts calibration against OT.

4. Pure text+chemistry without GNN is surprisingly strong on `cross_disease`.
   - `ablation_no_gnn` reaches the best `cross_disease` Pair AUROC (`0.8828`) and OT AUROC (`0.7236`).
   - This means current graph structure under strict clean is not helping disease OOD enough, and can even be a liability.

5. Path loss improves mechanism sensitivity but does not create a new overall winner.
   - `variant_path_loss` consistently improves `Pair-Fixed HO AUPRC` over `sota_dual_tower`.
   - But the gains are modest and do not recover the OT drop introduced by dual-tower fusion.

6. Pair-Fixed HO reveals true mechanism discrimination more faithfully than original HO Hit@1/MRR.
   - Under `ablation_no_gnn`, Pair-Fixed HO AUPRC collapses to approximately random (`0.3333`) across all splits.
   - This confirms that without graph features, the model loses real path-level discrimination even if some conventional HO ranking metrics appear inflated.

## Interpretation for Reporting

- If the paper wants to argue that graph topology remains indispensable, the current full `ablation_no_gnn` result weakens that claim for `cross_disease`.
- A more defensible statement is:
  - GNN contributes genuine path-level mechanism discrimination.
  - Disease text is the dominant source for zero-shot disease generalization.
  - Drug text and topology interact non-trivially, and stronger pairwise numbers do not necessarily imply better external calibration.

## Most Defensible Summary Statement

The current experiments suggest that under strict pair-level clean evaluation, disease-side semantic priors are the main source of zero-shot cross-disease performance, while the graph encoder primarily contributes mechanism-level discrimination rather than dominating pairwise AUROC. Drug-side semantic augmentation and auxiliary path loss improve some internal objectives but introduce a tradeoff against external OT generalization.

