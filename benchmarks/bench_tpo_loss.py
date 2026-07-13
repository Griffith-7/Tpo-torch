"""TPO loss benchmarks — correctness + performance."""

import time
import torch
import pytest
from tpo_torch.loss import tpo_loss_from_logits, tpo_loss


class TestTPOLossCorrectness:
    """Verify the TPO loss matches the paper formulation."""

    def test_loss_is_nonnegative(self):
        policy_logits = torch.randn(2, 8, 16, requires_grad=True)
        ref_logits = torch.randn(2, 8, 16)
        labels = torch.randint(0, 16, (2, 8))
        advantages = torch.tensor([1.0, -1.0])

        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.1)
        assert loss.item() >= 0, "Loss must be non-negative"

    def test_high_advantage_reduces_loss_for_good_token(self):
        torch.manual_seed(0)
        batch, seq, vocab = 1, 1, 10
        policy_logits = torch.zeros(batch, seq + 1, vocab, requires_grad=True)
        ref_logits = torch.zeros(batch, seq + 1, vocab)
        labels = torch.tensor([[5]])

        # High advantage should push target toward 1.0 for label token
        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, torch.tensor([5.0]), beta=0.1)
        loss.backward()
        grad = policy_logits.grad.clone()

        # Gradient at label position should be negative (increasing log-prob reduces loss)
        assert grad[0, 0, 5] < 0, "High advantage should increase log-prob of label token"

    def test_negative_advantage_decreases_prob(self):
        torch.manual_seed(0)
        batch, seq, vocab = 1, 1, 10
        policy_logits = torch.zeros(batch, seq + 1, vocab, requires_grad=True)
        ref_logits = torch.zeros(batch, seq + 1, vocab)
        labels = torch.tensor([[5]])

        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, torch.tensor([-5.0]), beta=0.1)
        loss.backward()
        grad = policy_logits.grad.clone()

        # Gradient at label position should be positive (decreasing log-prob increases loss)
        assert grad[0, 0, 5] > 0, "Negative advantage should decrease log-prob of label token"

    def test_beta_temperature_effect(self):
        torch.manual_seed(42)
        batch, seq, vocab = 1, 1, 10
        logits = torch.zeros(batch, seq + 1, vocab, requires_grad=True)
        ref = torch.zeros(batch, seq + 1, vocab)
        labels = torch.tensor([[5]])

        loss_small_beta = tpo_loss_from_logits(logits, ref, labels, torch.tensor([3.0]), beta=0.01)
        loss_large_beta = tpo_loss_from_logits(logits, ref, labels, torch.tensor([3.0]), beta=10.0)

        # Smaller beta = stronger push = larger loss magnitude
        assert loss_small_beta.item() != loss_large_beta.item(), "Beta should affect loss"

    def test_numerical_stability_extreme_advantages(self):
        policy_logits = torch.randn(4, 16, 32, requires_grad=True)
        ref_logits = torch.randn(4, 16, 32)
        labels = torch.randint(0, 32, (4, 16))
        advantages = torch.tensor([100.0, -100.0, 50.0, -50.0])

        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.01)
        assert not torch.isnan(loss), "Loss must not be NaN with extreme advantages"
        loss.backward()
        assert not torch.isnan(policy_logits.grad).any(), "Gradients must not be NaN"

    def test_masked_tokens_have_zero_gradient(self):
        policy_logits = torch.randn(2, 8, 16, requires_grad=True)
        ref_logits = torch.randn(2, 8, 16)
        labels = torch.randint(0, 16, (2, 8))
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float)

        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.5, attention_mask=mask)
        loss.backward()
        assert torch.all(policy_logits.grad[0, 4:] == 0), "Masked tokens must have zero gradient"
        assert torch.all(policy_logits.grad[1, 5:] == 0), "Masked tokens must have zero gradient"


class TestTPOLossPerformance:
    """Benchmark TPO loss computation speed."""

    def test_loss_speed(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch, seq, vocab = 8, 128, 32000

        policy_logits = torch.randn(batch, seq, vocab, device=device, requires_grad=True)
        ref_logits = torch.randn(batch, seq, vocab, device=device)
        labels = torch.randint(0, vocab, (batch, seq), device=device)
        advantages = torch.randn(batch, device=device)

        # Warmup
        for _ in range(3):
            loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages)
            loss.backward()
            policy_logits.grad = None

        n_iters = 20
        start = time.perf_counter()
        for _ in range(n_iters):
            loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages)
            loss.backward()
            policy_logits.grad = None
        elapsed = time.perf_counter() - start
        avg_ms = elapsed / n_iters * 1000

        print(f"\n  TPO loss ({device}): {avg_ms:.2f}ms/step (batch={batch}, seq={seq}, vocab={vocab})")
        assert avg_ms < 1000, f"Loss too slow: {avg_ms:.0f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
