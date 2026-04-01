from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'assets' / 'readme' / 'summary_metrics.json'
OUT_DIR = ROOT / 'assets' / 'readme'
SPLIT_ORDER = ['random', 'cross_drug', 'cross_disease']
FULL_COLOR = '#1f4e79'
TEXT_COLOR = '#9ecae1'

plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def load_payload() -> dict:
    return json.loads(DATA_PATH.read_text(encoding='utf-8-sig'))


def draw_overview(payload: dict) -> None:
    metrics = [
        ('pair_auroc', 'Internal Pair AUROC', (0.78, 1.0)),
        ('ot_auroc', 'OT AUROC', (0.78, 1.0)),
        ('ho_auprc', 'HO AUPRC', (0.28, 0.36)),
    ]
    split_payload = payload['full_multiseed']['splits']
    x = np.arange(len(SPLIT_ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    for ax, (metric_key, title, ylim) in zip(axes, metrics):
        means = [split_payload[split][metric_key][0] for split in SPLIT_ORDER]
        stds = [split_payload[split][metric_key][1] for split in SPLIT_ORDER]
        labels = [split_payload[split]['label'] for split in SPLIT_ORDER]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=['#4C78A8', '#72B7B2', '#F58518'])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.2)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.004, f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Latest v2 protocol results (3 seeds)', y=1.03, fontsize=13)
    fig.savefig(OUT_DIR / 'performance_overview.svg', format='svg')
    plt.close(fig)


def draw_text_vs_full(payload: dict) -> None:
    split_payload = payload['text_vs_full_seed42']['splits']
    metrics = [
        ('pair_auroc', 'Pair AUROC', (0.75, 1.0)),
        ('ot_auroc', 'OT AUROC', (0.75, 1.0)),
    ]
    x = np.arange(len(SPLIT_ORDER))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    for ax, (metric_key, title, ylim) in zip(axes, metrics):
        text_vals = [split_payload[split]['text_only'][metric_key] for split in SPLIT_ORDER]
        full_vals = [split_payload[split]['full'][metric_key] for split in SPLIT_ORDER]
        labels = [split_payload[split]['label'] for split in SPLIT_ORDER]
        ax.bar(x - width / 2, text_vals, width=width, color=TEXT_COLOR, label='Text-only')
        ax.bar(x + width / 2, full_vals, width=width, color=FULL_COLOR, label='Full model')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.2)
        for idx, (tval, fval) in enumerate(zip(text_vals, full_vals)):
            delta = fval - tval
            ax.text(x[idx], max(tval, fval) + 0.005, f'{delta:+.3f}', ha='center', va='bottom', fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.suptitle('What graph modules add on top of text-only features (seed=42)', y=1.08, fontsize=13)
    fig.savefig(OUT_DIR / 'text_vs_full.svg', format='svg')
    plt.close(fig)


def draw_cross_disease_focus(payload: dict) -> None:
    ablation = payload['cross_disease_ablation_seed42']
    order = ablation['order']
    variants = ablation['variants']
    labels = [variants[key]['label'] for key in order]
    colors = [variants[key]['color'] for key in order]
    metrics = [
        ('pair_auroc', 'Pair AUROC', (0.80, 0.93)),
        ('ot_auroc', 'OT AUROC', (0.80, 0.91)),
        ('ho_auprc', 'HO AUPRC', (0.30, 0.33)),
    ]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    for ax, (metric_key, title, ylim) in zip(axes, metrics):
        values = [variants[key][metric_key] for key in order]
        bars = ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha='right')
        ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Cross-disease ablation snapshot under the latest protocol (seed=42)', y=1.04, fontsize=13)
    fig.savefig(OUT_DIR / 'cross_disease_focus.svg', format='svg')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_payload()
    draw_overview(payload)
    draw_text_vs_full(payload)
    draw_cross_disease_focus(payload)
    print(f'Saved figures to {OUT_DIR}')


if __name__ == '__main__':
    main()
