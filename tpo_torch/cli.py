"""TPO-Torch CLI — train and evaluate TPO models from the command line."""

import argparse
import sys


def cmd_train(args: argparse.Namespace) -> None:
    """Run TPO training."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    from tpo_torch.trainer import TPOTrainer

    print(f"[*] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading policy model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)

    print(f"[*] Loading reference model: {args.model}")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.dataset:
        print(f"[*] Loading dataset: {args.dataset}")
        from datasets import load_dataset

        train_dataset = load_dataset(args.dataset, split=args.split)
    else:
        print("[*] Using synthetic dataset (no --dataset provided)")
        train_dataset = _create_synthetic_dataset(tokenizer, num_samples=args.num_samples)

    print(f"[*] Configuring training: {args.max_steps} steps, lr={args.lr}, beta={args.beta}")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        report_to=["none"],
        bf16=torch.cuda.is_available(),
    )

    trainer = TPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("[*] Starting TPO training...")
    result = trainer.train()
    print(f"[*] Training complete. Final loss: {result.training_loss:.4f}")
    print(f"[*] Model saved to: {args.output_dir}")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run TPO benchmarks."""
    print("[*] Running TPO loss benchmarks...")
    import time

    import torch

    from tpo_torch.loss import tpo_loss_from_logits

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    batch, seq, vocab = 8, 128, 32000
    policy_logits = torch.randn(batch, seq, vocab, device=device, requires_grad=True)
    ref_logits = torch.randn(batch, seq, vocab, device=device)
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    advantages = torch.randn(batch, device=device)

    # Warmup
    for _ in range(3):
        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.1)
        loss.backward()
        policy_logits.grad = None

    # Benchmark
    n_iters = 50
    start = time.perf_counter()
    for _ in range(n_iters):
        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.1)
        loss.backward()
        policy_logits.grad = None
    elapsed = time.perf_counter() - start

    print(f"[*] {n_iters} iterations in {elapsed:.3f}s")
    print(f"[*] Avg: {elapsed / n_iters * 1000:.2f}ms per step")
    print(f"[*] Loss value: {loss.item():.4f}")
    print("[*] Benchmark complete.")


def cmd_info(args: argparse.Namespace) -> None:
    """Show TPO package info."""
    from tpo_torch import __version__

    print(f"TPO-Torch v{__version__}")
    print("Target Policy Optimization — experimental RLHF implementation")
    print("")
    print("Based on: arXiv:2604.06159 (Kaddour, 2026)")
    print("License: MIT")
    print("")
    print("Commands:")
    print("  tpo train    Train a model with TPO")
    print("  tpo bench    Run TPO loss benchmarks")
    print("  tpo info     Show package info")


def _create_synthetic_dataset(tokenizer, num_samples: int = 100):
    """Create a synthetic dataset for quick testing."""
    import numpy as np
    from datasets import Dataset

    data = []
    for i in range(num_samples):
        text = f"Sample prompt {i}. What is reinforcement learning?"
        tokens = tokenizer(text, truncation=True, max_length=128, padding="max_length")
        data.append(
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": tokens["input_ids"],
                "advantages": float(np.random.uniform(0.1, 1.5)),
            }
        )
    return Dataset.from_list(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tpo",
        description="TPO-Torch: Target Policy Optimization for RLHF",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    p_train = subparsers.add_parser("train", help="Train a model with TPO")
    p_train.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model name"
    )
    p_train.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    p_train.add_argument("--split", type=str, default="train", help="Dataset split")
    p_train.add_argument("--output-dir", type=str, default="./tpo_output", help="Output directory")
    p_train.add_argument("--max-steps", type=int, default=100, help="Max training steps")
    p_train.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p_train.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p_train.add_argument("--beta", type=float, default=0.1, help="TPO beta temperature")
    p_train.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    p_train.add_argument(
        "--num-samples", type=int, default=100, help="Synthetic dataset size (if no dataset)"
    )
    p_train.set_defaults(func=cmd_train)

    # bench
    p_bench = subparsers.add_parser("bench", help="Run TPO loss benchmarks")
    p_bench.set_defaults(func=cmd_bench)

    # info
    p_info = subparsers.add_parser("info", help="Show package info")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
