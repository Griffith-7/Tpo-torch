"""TPO-Torch GPU Benchmark Suite — training curves, gradient stability, speed.

Generates:
  benchmarks/results/figures/training_curves.png
  benchmarks/results/figures/gradient_stability.png
  benchmarks/results/figures/speed_scaling.png
  benchmarks/results/benchmark_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tpo_torch.loss import tpo_loss_from_logits

FIG_DIR = Path(__file__).parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: Training curves — TPO vs vanilla cross-entropy
# ═════════════════════════════════════════════════════════════════════════════
def train_tiny(loss_fn, n_steps=40, lr=1e-3, device="cpu", seed=42):
    torch.manual_seed(seed)
    vocab, d, seq, batch = 128, 32, 8, 16
    embed = nn.Embedding(vocab, d).to(device)
    head = nn.Linear(d, vocab).to(device)
    params = list(embed.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=lr)

    losses = []
    for _ in range(n_steps):
        x = torch.randint(0, vocab, (batch, seq), device=device)
        labels = torch.randint(0, vocab, (batch, seq), device=device)
        h = embed(x)
        logits = head(h)

        if loss_fn == "tpo":
            ref = head(h).detach()
            advs = torch.randn(batch, device=device)
            loss = tpo_loss_from_logits(logits, ref, labels, advs, beta=0.1)
        else:
            loss = nn.functional.cross_entropy(logits.view(-1, vocab), labels.view(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return losses


def run_training(device):
    print("[1/3] Training curves — TPO vs CE ...")
    n_seeds = 3
    ce = [train_tiny("ce", device=device, seed=s) for s in range(n_seeds)]
    tpo = [train_tiny("tpo", device=device, seed=s) for s in range(n_seeds)]

    ce_m, ce_s = np.mean(ce, axis=0), np.std(ce, axis=0)
    tp_m, tp_s = np.mean(tpo, axis=0), np.std(tpo, axis=0)
    steps = np.arange(1, len(ce_m) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, ce_m, label="Cross-Entropy", color="#2196F3", lw=2)
    ax.fill_between(steps, ce_m - ce_s, ce_m + ce_s, alpha=0.15, color="#2196F3")
    ax.plot(steps, tp_m, label="TPO (\u03b2=0.1)", color="#E91E63", lw=2)
    ax.fill_between(steps, tp_m - tp_s, tp_m + tp_s, alpha=0.15, color="#E91E63")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("TPO vs Cross-Entropy: Training Loss Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'training_curves.png'}")
    return {"ce_final": round(float(ce_m[-1]), 4), "tpo_final": round(float(tp_m[-1]), 4)}


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: Gradient stability
# ═════════════════════════════════════════════════════════════════════════════
def run_stability(device):
    print("[2/3] Gradient stability ...")
    vocab, seq, batch = 256, 16, 8

    # Advantage sweep
    advs_vals = np.logspace(-2, 3, 15)
    grad_norms, nan_counts = [], []
    for av in advs_vals:
        nans, gns = 0, []
        for _ in range(5):
            logits = torch.randn(batch, seq, vocab, device=device, requires_grad=True)
            ref = torch.randn(batch, seq, vocab, device=device)
            labels = torch.randint(0, vocab, (batch, seq), device=device)
            loss = tpo_loss_from_logits(logits, ref, labels, torch.full((batch,), av, device=device), beta=0.1)
            loss.backward()
            if torch.isnan(loss):
                nans += 1
            elif logits.grad is not None:
                gns.append(logits.grad.norm().item())
        grad_norms.append(np.mean(gns) if gns else 0)
        nan_counts.append(nans)

    # Beta sweep
    betas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    beta_losses = []
    for b in betas:
        bl = []
        for _ in range(5):
            logits = torch.randn(batch, seq, vocab, device=device, requires_grad=True)
            ref = torch.randn(batch, seq, vocab, device=device)
            labels = torch.randint(0, vocab, (batch, seq), device=device)
            loss = tpo_loss_from_logits(logits, ref, labels, torch.randn(batch, device=device) * 2, beta=b)
            loss.backward()
            if not torch.isnan(loss):
                bl.append(loss.item())
        beta_losses.append(np.mean(bl) if bl else 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].semilogx(advs_vals, grad_norms, "o-", color="#4CAF50", lw=2, ms=4)
    axes[0].set_xlabel("Advantage Value")
    axes[0].set_ylabel("Mean Gradient Norm")
    axes[0].set_title("Gradient Magnitude vs Advantage")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(advs_vals, nan_counts, "s-", color="#F44336", lw=2, ms=4)
    axes[1].set_xlabel("Advantage Value")
    axes[1].set_ylabel("NaN Count (/5)")
    axes[1].set_title("Numerical Stability")
    axes[1].set_ylim(-0.5, 5.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(betas, beta_losses, "D-", color="#FF9800", lw=2, ms=6)
    axes[2].set_xlabel("Beta (temperature)")
    axes[2].set_ylabel("Mean Loss")
    axes[2].set_title("TPO Loss vs Beta")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("TPO Gradient Stability Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gradient_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'gradient_stability.png'}")
    return {
        "nan_count_at_extreme_adv": int(nan_counts[-1]),
        "max_grad_norm": round(float(max(grad_norms)), 4),
        "nan_free": int(nan_counts[-1]) == 0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PART 3: Speed scaling on GPU
# ═════════════════════════════════════════════════════════════════════════════
def run_speed(device):
    print(f"[3/3] Speed scaling on {device.upper()} ...")
    batch, vocab = 8, 32000
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    n_iters = 20
    times_ms, throughput = [], []

    for seq in seq_lengths:
        logits = torch.randn(batch, seq, vocab, device=device, requires_grad=True)
        ref = torch.randn(batch, seq, vocab, device=device)
        labels = torch.randint(0, vocab, (batch, seq), device=device)
        advs = torch.randn(batch, device=device)

        try:
            for _ in range(3):
                loss = tpo_loss_from_logits(logits, ref, labels, advs, beta=0.1)
                loss.backward()
                logits.grad = None

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_iters):
                loss = tpo_loss_from_logits(logits, ref, labels, advs, beta=0.1)
                loss.backward()
                logits.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / n_iters * 1000
            tok_s = batch * seq / (elapsed / 1000)

            times_ms.append(elapsed)
            throughput.append(tok_s)
            print(f"    seq={seq:>5}  {elapsed:>8.2f}ms  {tok_s:>10.0f} tok/s")
        except torch.cuda.OutOfMemoryError:
            print(f"    seq={seq:>5}  OOM — skipping larger sequences")
            torch.cuda.empty_cache()
            break
        finally:
            del logits, ref, labels, advs
            torch.cuda.empty_cache()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(seq_lengths, times_ms, "o-", color="#2196F3", lw=2, label="Latency (ms)")
    ax2.plot(seq_lengths, throughput, "s--", color="#E91E63", lw=2, label="Throughput (tok/s)")
    ax1.set_xlabel("Sequence Length", fontsize=12)
    ax1.set_ylabel("Latency per Step (ms)", fontsize=12, color="#2196F3")
    ax2.set_ylabel("Throughput (tokens/sec)", fontsize=12, color="#E91E63")
    ax1.set_title(f"TPO Loss Speed \u2014 batch={batch}, vocab={vocab}, {device.upper()}", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2.tick_params(axis="y", labelcolor="#E91E63")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speed_scaling.png", dpi=150)
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'speed_scaling.png'}")

    return {
        "device": device,
        "results": [
            {"seq_len": s, "latency_ms": round(t, 2), "throughput_tok_s": round(tp)}
            for s, t, tp in zip(seq_lengths, times_ms, throughput)
        ],
    }


def main():
    print("=" * 60)
    print("  TPO-Torch Benchmark Suite")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    results = {}
    results["training"] = run_training(device)
    print()
    results["stability"] = run_stability(device)
    print()
    results["speed"] = run_speed(device)

    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {RESULTS_DIR / 'benchmark_results.json'}")
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
