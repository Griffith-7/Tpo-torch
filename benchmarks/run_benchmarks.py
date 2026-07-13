"""TPO-Torch Benchmark — TPO vs PPO (with CE as reference).

TPO competes with PPO, not cross-entropy. This benchmark compares
TPO and PPO on the same metric (held-out perplexity). CE is shown
as a supervised-learning baseline for reference.

Usage:
    python benchmarks/run_benchmarks.py
    # Graphs: benchmarks/results/figures/
    # JSON:   benchmarks/results/benchmark_results.json
"""

import json
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FIG_DIR = Path(__file__).parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"


def tpo_loss(policy_logits, reference_logits, labels, advantages, beta=0.1,
             attention_mask=None):
    """TPO loss from logits — inlined to avoid CUDA crash from __init__ import chain."""
    shift_p = policy_logits[..., :-1, :].contiguous()
    shift_r = reference_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    seq = shift_p.size(1)

    p_lp = F.log_softmax(shift_p, dim=-1)
    r_lp = F.log_softmax(shift_r, dim=-1)
    p_g = torch.gather(p_lp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    r_g = torch.gather(r_lp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1).expand(-1, seq)

    with torch.no_grad():
        p_ref = torch.exp(r_g).clamp(min=1e-8, max=1 - 1e-8)
        log_odds = torch.log(p_ref) - torch.log(1.0 - p_ref)
        target = torch.sigmoid(log_odds + advantages / beta)

    per_token_loss = -(target * p_g)

    if attention_mask is not None:
        mask = attention_mask[..., 1:].float()
        loss = (per_token_loss * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = per_token_loss.mean()
    return loss


class TinyTransformer(nn.Module):
    def __init__(self, vocab=64, d=64, seq=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        enc = nn.TransformerEncoderLayer(
            d, nhead=4, dim_feedforward=d * 2, batch_first=True
        )
        self.tf = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        return self.head(self.tf(self.embed(x) + self.pos(pos)))


def make_data(n, seq=16, vocab=64):
    x = torch.randint(1, vocab, (n, seq))
    return x, torch.cat([x[:, 1:], x[:, :1]], dim=1)


def ppl(model, vx, vy, vocab=64):
    model.eval()
    with torch.no_grad():
        loss = F.cross_entropy(model(vx).view(-1, vocab), vy.view(-1))
    model.train()
    return torch.exp(loss).item()


def train_ce(model, tx, ty, vx, vy, vocab=64):
    """CE baseline — supervised learning reference."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ppls = []
    for _ in range(60):
        model.train()
        logits = model(tx)
        loss = F.cross_entropy(logits.view(-1, vocab), ty.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        ppls.append(ppl(model, vx, vy, vocab))
    return ppls


def train_ppo(model, tx, ty, vx, vy, vocab=64):
    """PPO-clip (simplified, no value function, no GAE)."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ppls = []
    for _ in range(60):
        model.train()
        logits = model(tx)
        with torch.no_grad():
            old_lp = (
                F.log_softmax(logits, -1)
                .gather(2, ty.unsqueeze(-1))
                .squeeze(-1)
            )
            rewards = (logits.argmax(-1) == ty).float()
        new_lp = (
            F.log_softmax(logits, -1).gather(2, ty.unsqueeze(-1)).squeeze(-1)
        )
        ratio = torch.exp(new_lp - old_lp)
        adv = rewards - 0.5
        loss = -torch.min(
            ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv
        ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ppls.append(ppl(model, vx, vy, vocab))
    return ppls


def train_tpo(model, tx, ty, vx, vy, vocab=64):
    """TPO — loss function only, no extra infrastructure."""
    ref = TinyTransformer(vocab=vocab)
    ref.load_state_dict(
        {k: v.clone() for k, v in model.state_dict().items()}
    )
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ppls = []
    for _ in range(60):
        model.train()
        logits = model(tx)
        with torch.no_grad():
            advs = (logits.argmax(-1) == ty).float().mean(1) - 0.5
            ref_logits = ref(tx)
        loss = tpo_loss(logits, ref_logits, ty, advs, beta=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ppls.append(ppl(model, vx, vy, vocab))
    return ppls


def run_training(vocab=64, seq=16, n_seeds=3):
    print(f"[1/3] Training: TPO vs PPO-clip vs CE baseline ({n_seeds} seeds)")
    tx, ty = make_data(256, seq, vocab)
    vx, vy = make_data(64, seq, vocab)

    all_ppls = {}
    for name, fn in [
        ("CE (baseline)", train_ce),
        ("PPO-clip", train_ppo),
        ("TPO", train_tpo),
    ]:
        ppls = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            m = TinyTransformer(vocab, 64, seq)
            ppls.append(fn(m, tx, ty, vx, vy, vocab))
            print(f"  {name} seed {seed}: final={ppls[-1][-1]:.2f}")
        all_ppls[name] = ppls

    means = {k: np.mean(v, axis=0) for k, v in all_ppls.items()}
    stds = {k: np.std(v, axis=0) for k, v in all_ppls.items()}
    steps = np.arange(1, len(means["TPO"]) + 1)

    colors = {
        "CE (baseline)": "#9E9E9E",
        "PPO-clip": "#4CAF50",
        "TPO": "#E91E63",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ["CE (baseline)", "PPO-clip", "TPO"]:
        ax.plot(steps, means[name], label=name, color=colors[name], lw=2)
        ax.fill_between(
            steps,
            means[name] - stds[name],
            means[name] + stds[name],
            alpha=0.1,
            color=colors[name],
        )
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Held-out Perplexity (lower = better)", fontsize=12)
    ax.set_title("TPO vs PPO-clip: Held-out Perplexity", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'training_curves.png'}")
    return {
        "metric": "perplexity_on_held_out_data",
        "ce_baseline": round(float(means["CE (baseline)"][-1]), 2),
        "ppo_final": round(float(means["PPO-clip"][-1]), 2),
        "tpo_final": round(float(means["TPO"][-1]), 2),
    }


def run_stability():
    print("[2/3] Gradient stability ...")
    batch_size, seq, vocab = 8, 16, 256
    advs_vals = np.logspace(-2, 3, 15)
    grad_norms, nan_counts = [], []
    for av in advs_vals:
        nans, gns = 0, []
        for _ in range(5):
            logits = torch.randn(
                batch_size, seq, vocab, requires_grad=True
            )
            ref = torch.randn(batch_size, seq, vocab)
            labels = torch.randint(0, vocab, (batch_size, seq))
            loss = tpo_loss(
                logits, ref, labels,
                torch.full((batch_size,), av), beta=0.1,
            )
            loss.backward()
            nans += int(torch.isnan(loss))
            if logits.grad is not None:
                gns.append(logits.grad.norm().item())
        grad_norms.append(np.mean(gns) if gns else 0)
        nan_counts.append(nans)

    betas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    beta_losses = []
    for b in betas:
        bl = []
        for _ in range(5):
            logits = torch.randn(
                batch_size, seq, vocab, requires_grad=True
            )
            ref = torch.randn(batch_size, seq, vocab)
            labels = torch.randint(0, vocab, (batch_size, seq))
            loss = tpo_loss(
                logits, ref, labels,
                torch.randn(batch_size) * 2, beta=b,
            )
            if not torch.isnan(loss):
                bl.append(loss.item())
        beta_losses.append(np.mean(bl) if bl else 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].semilogx(
        advs_vals, grad_norms, "o-", color="#4CAF50", lw=2, ms=4
    )
    axes[0].set_xlabel("Advantage Value")
    axes[0].set_ylabel("Mean Gradient Norm")
    axes[0].set_title("Gradient Magnitude vs Advantage")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(
        advs_vals, nan_counts, "s-", color="#F44336", lw=2, ms=4
    )
    axes[1].set_xlabel("Advantage Value")
    axes[1].set_ylabel("NaN Count (/5)")
    axes[1].set_title("Numerical Stability (0 = good)")
    axes[1].set_ylim(-0.5, 5.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(
        betas, beta_losses, "D-", color="#FF9800", lw=2, ms=6
    )
    axes[2].set_xlabel("Beta (temperature)")
    axes[2].set_ylabel("Mean Loss")
    axes[2].set_title("TPO Loss vs Beta")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("TPO Gradient Stability", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "gradient_stability.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'gradient_stability.png'}")
    return {
        "nan_free": int(nan_counts[-1]) == 0,
        "max_grad_norm": round(float(max(grad_norms)), 4),
        "advantage_range": [float(advs_vals[0]), float(advs_vals[-1])],
    }


def run_speed():
    """Speed test — imports tpo_torch if available, otherwise skips."""
    print("[3/3] Speed scaling ...")
    try:
        from tpo_torch.loss import tpo_loss_from_logits
    except Exception as e:
        print(f"  Skipped (import failed: {e})")
        return {"skipped": True, "reason": str(e)}

    batch, vocab = 8, 32000
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    times_ms, throughput = [], []
    for seq_len in seq_lengths:
        try:
            logits = torch.randn(
                batch, seq_len, vocab, device=device, requires_grad=True
            )
            ref = torch.randn(batch, seq_len, vocab, device=device)
            labels = torch.randint(0, vocab, (batch, seq_len), device=device)
            advs = torch.randn(batch, device=device)

            for _ in range(3):
                loss = tpo_loss_from_logits(
                    logits, ref, labels, advs, beta=0.1
                )
                loss.backward()
                logits.grad = None
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(20):
                loss = tpo_loss_from_logits(
                    logits, ref, labels, advs, beta=0.1
                )
                loss.backward()
                logits.grad = None
            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) / 20 * 1000
            tok_s = batch * seq_len / (elapsed / 1000)
            times_ms.append(elapsed)
            throughput.append(tok_s)
            print(
                f"    seq={seq_len:>5}  {elapsed:>8.2f}ms"
                f"  {tok_s:>10.0f} tok/s"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        finally:
            try:
                del logits, ref, labels, advs
            except Exception:
                pass
            torch.cuda.empty_cache()

    if not times_ms:
        return {"skipped": True, "reason": "all seq lengths failed"}

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(
        seq_lengths[: len(times_ms)], times_ms,
        "o-", color="#2196F3", lw=2, label="Latency (ms)",
    )
    ax2.plot(
        seq_lengths[: len(throughput)], throughput,
        "s--", color="#E91E63", lw=2, label="Throughput (tok/s)",
    )
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency per Step (ms)", color="#2196F3")
    ax2.set_ylabel("Throughput (tokens/sec)", color="#E91E63")
    ax1.set_title(
        f"TPO Loss Speed — batch={batch}, vocab={vocab}, {device.upper()}"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speed_scaling.png", dpi=150)
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'speed_scaling.png'}")
    return {
        "device": device,
        "results": [
            {
                "seq_len": s,
                "latency_ms": round(t, 2),
                "throughput_tok_s": round(tp),
            }
            for s, t, tp in zip(
                seq_lengths[: len(times_ms)], times_ms, throughput
            )
        ],
    }


def main():
    print("=" * 60)
    print("  TPO-Torch Benchmark")
    print("  TPO competes with PPO, not cross-entropy.")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    results = {}
    results["training"] = run_training()
    print()
    results["stability"] = run_stability()
    print()
    results["speed"] = run_speed()

    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results: {RESULTS_DIR / 'benchmark_results.json'}")
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
