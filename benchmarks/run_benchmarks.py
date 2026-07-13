"""TPO-Torch Honest Benchmark — TPO vs CE vs PPO-clip on the same task.

All methods trained on the same synthetic task (token identity prediction).
All methods evaluated on the SAME metric: held-out perplexity.
No loss-value comparisons across different loss functions.
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
import torch.nn.functional as F

FIG_DIR = Path(__file__).parent / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"


# ═════════════════════════════════════════════════════════════════════════════
# Model: tiny 2-layer transformer
# ═════════════════════════════════════════════════════════════════════════════
class TinyTransformer(nn.Module):
    def __init__(self, vocab=64, d=64, n_heads=4, n_layers=2, seq_len=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos(pos)
        h = self.transformer(h)
        return self.head(h)


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic task: predict the NEXT token (identity shift)
# Labels[i] = input[i+1], last label = input[0] (wrap around)
# ═════════════════════════════════════════════════════════════════════════════
def make_data(n_samples, seq_len, vocab, device):
    inputs = torch.randint(1, vocab, (n_samples, seq_len), device=device)
    labels = torch.cat([inputs[:, 1:], inputs[:, :1]], dim=1)
    return inputs, labels


def compute_perplexity(model, inputs, labels):
    """Evaluate perplexity on held-out data — same metric for ALL methods."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    model.train()
    return torch.exp(loss).item()


# ═════════════════════════════════════════════════════════════════════════════
# Training methods
# ═════════════════════════════════════════════════════════════════════════════
def train_ce(model, train_x, train_y, val_x, val_y, n_steps, lr, device):
    """Standard cross-entropy training (baseline)."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    ppls = []
    for step in range(n_steps):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), train_y.view(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        ppls.append(compute_perplexity(model, val_x, val_y))
    return ppls


def train_ppo_clip(model, train_x, train_y, val_x, val_y, n_steps, lr, device, clip_eps=0.2):
    """PPO-clip style training (simplified, no value function)."""
    from tpo_torch.loss import tpo_loss_from_logits

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    ppls = []
    for step in range(n_steps):
        model.train()
        logits = model(train_x)

        # Compute old log-probs for importance sampling
        with torch.no_grad():
            old_logprobs = F.log_softmax(logits, dim=-1)
            old_lp = torch.gather(old_logprobs, dim=-1, index=train_y.unsqueeze(-1)).squeeze(-1)

        # Compute advantages from reward (per-token accuracy as reward)
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            token_rewards = (preds == train_y).float()

        # PPO-clip surrogate loss
        logprobs = F.log_softmax(logits, dim=-1)
        new_lp = torch.gather(logprobs, dim=-1, index=train_y.unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(new_lp - old_lp)
        advantages = token_rewards - 0.5  # centered reward

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        ppls.append(compute_perplexity(model, val_x, val_y))
    return ppls


def train_tpo(model, train_x, train_y, val_x, val_y, n_steps, lr, device, beta=0.1):
    """TPO training with advantage-weighted target distributions."""
    from tpo_torch.loss import tpo_loss_from_logits

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    ppls = []

    # Frozen reference (copy of initial weights)
    ref_model = type(model)(**{k: v for k, v in model.named_parameters().__class__.__dict__.items() if False})
    ref_model = type(model)(vocab=model.embed.num_embeddings, d=model.embed.embedding_dim,
                            n_heads=4, n_layers=2, seq_len=train_x.size(1))
    ref_model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    for step in range(n_steps):
        model.train()
        logits = model(train_x)

        # Compute advantages from per-token accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            per_token_reward = (preds == train_y).float()
            advantages = per_token_reward.mean(dim=1) - 0.5  # centered

        with torch.no_grad():
            ref_logits = ref_model(train_x)

        loss = tpo_loss_from_logits(
            policy_logits=logits,
            reference_logits=ref_logits,
            labels=train_y,
            advantages=advantages,
            beta=beta,
        )

        optim.zero_grad()
        loss.backward()
        optim.step()
        ppls.append(compute_perplexity(model, val_x, val_y))
    return ppls


# ═════════════════════════════════════════════════════════════════════════════
# Gradient stability (valid test — same as before, but with fixed math)
# ═════════════════════════════════════════════════════════════════════════════
def run_stability(device):
    print("[2/3] Gradient stability ...")
    from tpo_torch.loss import tpo_loss_from_logits

    vocab, seq, batch = 256, 16, 8

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
    axes[1].set_title("Numerical Stability (0 = good)")
    axes[1].set_ylim(-0.5, 5.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(betas, beta_losses, "D-", color="#FF9800", lw=2, ms=6)
    axes[2].set_xlabel("Beta (temperature)")
    axes[2].set_ylabel("Mean Loss")
    axes[2].set_title("TPO Loss vs Beta")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("TPO Gradient Stability (corrected math)", fontsize=15, y=1.02)
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
# Speed scaling
# ═════════════════════════════════════════════════════════════════════════════
def run_speed(device):
    print(f"[3/3] Speed scaling on {device.upper()} ...")
    from tpo_torch.loss import tpo_loss_from_logits

    batch, vocab = 8, 32000
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    n_iters = 20
    times_ms, throughput = [], []

    for seq in seq_lengths:
        try:
            logits = torch.randn(batch, seq, vocab, device=device, requires_grad=True)
            ref = torch.randn(batch, seq, vocab, device=device)
            labels = torch.randint(0, vocab, (batch, seq), device=device)
            advs = torch.randn(batch, device=device)

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
            print(f"    seq={seq:>5}  OOM — stopping")
            torch.cuda.empty_cache()
            break
        finally:
            del logits, ref, labels, advs
            torch.cuda.empty_cache()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(seq_lengths[:len(times_ms)], times_ms, "o-", color="#2196F3", lw=2, label="Latency (ms)")
    ax2.plot(seq_lengths[:len(throughput)], throughput, "s--", color="#E91E63", lw=2, label="Throughput (tok/s)")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency per Step (ms)", color="#2196F3")
    ax2.set_ylabel("Throughput (tokens/sec)", color="#E91E63")
    ax1.set_title(f"TPO Loss Speed \u2014 batch={batch}, vocab={vocab}, {device.upper()}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "speed_scaling.png", dpi=150)
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'speed_scaling.png'}")
    return {"device": device, "results": [
        {"seq_len": s, "latency_ms": round(t, 2), "throughput_tok_s": round(tp)}
        for s, t, tp in zip(seq_lengths[:len(times_ms)], times_ms, throughput)
    ]}


# ═════════════════════════════════════════════════════════════════════════════
# Main: run all benchmarks
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  TPO-Torch Benchmark Suite (corrected math)")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    results = {}

    # ── Part 1: Training comparison on PERPLEXITY ──
    print("[1/3] Training comparison: TPO vs CE vs PPO-clip (eval: perplexity) ...")

    vocab, seq_len, n_steps = 64, 16, 60
    train_x, train_y = make_data(256, seq_len, vocab, device)
    val_x, val_y = make_data(64, seq_len, vocab, device)

    n_seeds = 3
    ce_ppls, ppo_ppls, tpo_ppls = [], [], []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        m_ce = TinyTransformer(vocab=vocab, seq_len=seq_len).to(device)
        m_ppo = TinyTransformer(vocab=vocab, seq_len=seq_len).to(device)
        m_tpo = TinyTransformer(vocab=vocab, seq_len=seq_len).to(device)

        ce_ppls.append(train_ce(m_ce, train_x, train_y, val_x, val_y, n_steps, 3e-4, device))
        ppo_ppls.append(train_ppo_clip(m_ppo, train_x, train_y, val_x, val_y, n_steps, 3e-4, device))
        tpo_ppls.append(train_tpo(m_tpo, train_x, train_y, val_x, val_y, n_steps, 3e-4, device))
        print(f"  seed {seed}: CE={ce_ppls[-1][-1]:.2f} PPO={ppo_ppls[-1][-1]:.2f} TPO={tpo_ppls[-1][-1]:.2f}")

    ce_m = np.mean(ce_ppls, axis=0)
    ce_s = np.std(ce_ppls, axis=0)
    ppo_m = np.mean(ppo_ppls, axis=0)
    ppo_s = np.std(ppo_ppls, axis=0)
    tpo_m = np.mean(tpo_ppls, axis=0)
    tpo_s = np.std(tpo_ppls, axis=0)
    steps = np.arange(1, len(ce_m) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, ce_m, label="Cross-Entropy", color="#2196F3", lw=2)
    ax.fill_between(steps, ce_m - ce_s, ce_m + ce_s, alpha=0.1, color="#2196F3")
    ax.plot(steps, ppo_m, label="PPO-clip", color="#4CAF50", lw=2)
    ax.fill_between(steps, ppo_m - ppo_s, ppo_m + ppo_s, alpha=0.1, color="#4CAF50")
    ax.plot(steps, tpo_m, label="TPO (\u03b2=0.1)", color="#E91E63", lw=2)
    ax.fill_between(steps, tpo_m - tpo_s, tpo_m + tpo_s, alpha=0.1, color="#E91E63")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Held-out Perplexity (lower = better)", fontsize=12)
    ax.set_title("TPO vs CE vs PPO-clip: Perplexity on Held-out Data", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"    -> {FIG_DIR / 'training_curves.png'}")
    results["training"] = {
        "metric": "perplexity_on_held_out_data",
        "ce_final": round(float(ce_m[-1]), 2),
        "ppo_final": round(float(ppo_m[-1]), 2),
        "tpo_final": round(float(tpo_m[-1]), 2),
        "ce_initial": round(float(ce_m[0]), 2),
        "tpo_initial": round(float(tpo_m[0]), 2),
    }
    print()

    # ── Part 2: Gradient stability ──
    results["stability"] = run_stability(device)
    print()

    # ── Part 3: Speed ──
    results["speed"] = run_speed(device)

    # Save
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {RESULTS_DIR / 'benchmark_results.json'}")
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
