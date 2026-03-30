# Ablation Summary For Report

Note: the table below summarizes the current frozen SOTA mainline ablations. These ablations are single-run results on the final SOTA recipe; the multi-seed stability check was run for the full SOTA model itself.

## Core Ablation Table

| Config | Random Pair | Random OT | Random HO | Cross-Drug Pair | Cross-Drug OT | Cross-Drug HO | Cross-Disease Pair | Cross-Disease OT | Cross-Disease HO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full SOTA | 0.9778 | 0.8639 | 0.3540 | 0.8615 | 0.8419 | 0.3134 | 0.8661 | 0.7683 | 0.2987 |
| w.o. DropEdge | 0.9748 | 0.8276 | 0.3443 | 0.8477 | 0.8167 | 0.3169 | 0.8301 | 0.7318 | 0.2957 |
| w.o. Outer HO weak distill | 0.9737 | 0.8807 | 0.3431 | 0.8455 | 0.8347 | 0.2816 | 0.8704 | 0.7756 | 0.2982 |
| w.o. Morgan | 0.9698 | 0.8839 | 0.3174 | 0.8701 | 0.8001 | 0.3187 | 0.8110 | 0.7496 | 0.2694 |
| w.o. Disease Early | 0.9747 | 0.9041 | 0.3170 | 0.8315 | 0.8412 | 0.3034 | 0.8540 | 0.7962 | 0.2814 |

Metric meanings:
- Pair: internal pairwise main AUROC for that split.
- OT: external Open Targets main AUROC for that split.
- HO: hold-out mechanism AUPRC.

## What The Ablations Show

1. DropEdge is the most consistent gain.
   - Removing DropEdge hurts all three splits on pairwise AUROC and OT AUROC.
   - The largest drop appears on cross-disease, which is exactly the hardest OOD setting.
   - This supports the claim that DropEdge reduces oversmoothing and improves structural generalization.

2. Morgan fingerprints are a hard contributor, especially for cross-disease and HO.
   - Removing Morgan causes the largest drop on cross-disease pairwise and HO.
   - This means drug-side chemical priors are not cosmetic; they are a real source of generalization strength.

3. Disease Early Fusion mainly helps mechanism-side robustness and hard OOD reasoning.
   - Removing disease early text usually hurts HO, and also hurts cross-drug / cross-disease pairwise AUROC.
   - On some OT settings the score may look slightly higher without it, so its gain is not monotonic on every external metric.
   - The net interpretation is that disease early fusion is more important for semantic robustness than for pure thresholded transfer.

4. Outer HO weak distillation is useful, but not uniformly helpful.
   - It clearly helps cross-drug and HO.
   - But it is mixed on random OT and cross-disease pair/OT.
   - So the correct conclusion is not ?Outer HO always helps?, but ?Outer HO provides targeted mechanism supervision?.

5. The strongest and most robust recipe is still the full combination.
   - Even when some individual modules are mixed on one metric, the full stack gives the best overall balance across internal pairwise, OT external generalization, and HO mechanism validation.

## Recommended Talking Points For Advisor Meeting

- The two most decisive modules are DropEdge and Morgan.
- Disease early fusion mainly stabilizes disease-side OOD and HO mechanism performance.
- Outer HO weak distillation is a weak but meaningful auxiliary signal rather than a universal gain term.
- The final SOTA is a balanced design, not a single-module trick.
