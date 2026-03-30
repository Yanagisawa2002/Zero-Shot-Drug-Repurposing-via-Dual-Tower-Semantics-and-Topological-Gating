from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'assets' / 'readme' / 'summary_metrics.json'
OUT_DIR = ROOT / 'assets' / 'readme'

VARIANT_ORDER = [
    'baseline_pure_gnn',
    'single_tower',
    'sota_dual_tower',
    'ablation_no_gnn',
    'variant_path_loss',
]
SPLIT_ORDER = ['random', 'cross_drug', 'cross_disease']
METRICS = [
    ('pair_auroc', 'Internal Pair AUROC'),
    ('ot_auroc', 'External OT AUROC'),
    ('ho_auprc', 'HO AUPRC'),
]

plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def load_payload() -> dict:
    return json.loads(DATA_PATH.read_text(encoding='utf-8'))


def draw_overview(payload: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(SPLIT_ORDER))
    width = 0.15

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        for idx, variant in enumerate(VARIANT_ORDER):
            means = [payload['splits'][split][variant][metric_key][0] for split in SPLIT_ORDER]
            stds = [payload['splits'][split][variant][metric_key][1] for split in SPLIT_ORDER]
            label = payload['variants'][variant]['label']
            color = payload['variants'][variant]['color']
            ax.bar(x + (idx - 2) * width, means, width=width, color=color, label=label, yerr=stds, capsize=2)

        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(['Random', 'Cross-Drug', 'Cross-Disease'])
        ax.grid(axis='y', alpha=0.2)
        if metric_key == 'ho_auprc':
            ax.set_ylim(0.20, 0.40)
        else:
            ax.set_ylim(0.60, 1.00)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.suptitle('Multi-seed comparison under strict pair-level clean', y=1.16, fontsize=13)
    fig.savefig(OUT_DIR / 'performance_overview.svg', format='svg')
    plt.close(fig)


def draw_cross_disease_focus(payload: dict) -> None:
    split = 'cross_disease'
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    labels = [payload['variants'][variant]['label'] for variant in VARIANT_ORDER]
    colors = [payload['variants'][variant]['color'] for variant in VARIANT_ORDER]
    x = np.arange(len(VARIANT_ORDER))

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        means = [payload['splits'][split][variant][metric_key][0] for variant in VARIANT_ORDER]
        stds = [payload['splits'][split][variant][metric_key][1] for variant in VARIANT_ORDER]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3)
        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.grid(axis='y', alpha=0.2)
        if metric_key == 'ho_auprc':
            ax.set_ylim(0.20, 0.36)
        else:
            ax.set_ylim(0.60, 0.92)

    fig.suptitle('Cross-disease zero-shot breakdown', y=1.05, fontsize=13)
    fig.savefig(OUT_DIR / 'cross_disease_focus.svg', format='svg')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_payload()
    draw_overview(payload)
    draw_cross_disease_focus(payload)
    print(f'Saved figures to {OUT_DIR}')


if __name__ == '__main__':
    main()
