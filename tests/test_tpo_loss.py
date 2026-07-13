import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tpo_torch.loss import tpo_loss, tpo_loss_from_logits


class TestTPOLossCore:
    def test_no_nan_on_extreme_advantages(self):
        batch, seq, vocab = 4, 10, 32
        policy_logits = torch.randn(batch, seq, vocab, requires_grad=True)
        ref_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        advantages = torch.tensor([10.0, -10.0, 100.0, -100.0])

        loss = tpo_loss_from_logits(policy_logits, ref_logits, labels, advantages, beta=0.1)
        assert not torch.isnan(loss), "TPO Loss exploded into NaN!"
        loss.backward()
        assert not torch.isnan(policy_logits.grad).any(), "Gradients resulted in NaN!"

    def test_gradient_proportional_scaling(self):
        torch.manual_seed(42)
        batch, seq, vocab = 1, 1, 10
        policy_logits = torch.zeros(batch, seq + 1, vocab, requires_grad=True)
        ref_logits = torch.zeros(batch, seq + 1, vocab)
        labels = torch.tensor([[5, 5]])

        loss_zero = tpo_loss_from_logits(
            policy_logits, ref_logits, labels, torch.tensor([0.0]), beta=1.0
        )
        loss_zero.backward()
        grad_zero = policy_logits.grad.clone()
        policy_logits.grad.zero_()

        loss_high = tpo_loss_from_logits(
            policy_logits, ref_logits, labels, torch.tensor([5.0]), beta=1.0
        )
        loss_high.backward()
        grad_high = policy_logits.grad.clone()
        policy_logits.grad.zero_()

        loss_neg = tpo_loss_from_logits(
            policy_logits, ref_logits, labels, torch.tensor([-5.0]), beta=1.0
        )
        loss_neg.backward()
        grad_neg = policy_logits.grad.clone()

        sum_high = grad_high.abs().sum().item()
        sum_zero = grad_zero.abs().sum().item()
        sum_neg = grad_neg.abs().sum().item()

        assert sum_high > sum_zero
        assert sum_zero > sum_neg
        assert grad_high[0, 0, 5] < 0
        assert grad_high[0, 0, 5] < grad_zero[0, 0, 5]

    def test_tpo_loss_from_precomputed_logprobs(self):
        torch.manual_seed(7)
        policy_lp = torch.randn(2, 8).log_softmax(-1)
        ref_lp = torch.randn(2, 8).log_softmax(-1)
        advantages = torch.tensor([1.0, -1.0])

        loss = tpo_loss(policy_lp, ref_lp, advantages, beta=0.1)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_tpo_loss_1d_input(self):
        policy_lp = torch.randn(4).log_softmax(0)
        ref_lp = torch.randn(4).log_softmax(0)
        advantages = torch.tensor([1.0])
        loss = tpo_loss(policy_lp, ref_lp, advantages, beta=0.1)
        assert not torch.isnan(loss)


class TestTPOMasking:
    def test_padding_masked_out_of_loss(self):
        batch, seq, vocab = 2, 8, 16
        policy_logits = torch.randn(batch, seq, vocab, requires_grad=True)
        ref_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        advantages = torch.tensor([1.0, 2.0])
        mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float
        )

        loss_with_mask = tpo_loss_from_logits(
            policy_logits, ref_logits, labels, advantages, beta=0.5, attention_mask=mask
        )
        loss_with_mask.backward()
        assert torch.all(policy_logits.grad[0, 4:] == 0), "Masked tokens must have ZERO gradient"


class TestTPODataCollator:
    def test_collator_preserves_advantages(self):
        from tpo_torch.trainer import TPODataCollator

        class MockTokenizer:
            pad_token_id = 0
            pad_token = "<pad>"

            def pad(self, encoded_inputs, padding=True, max_length=None,
                    pad_to_multiple_of=None, return_tensors="pt"):
                from torch.nn.utils.rnn import pad_sequence
                input_ids = [torch.tensor(x["input_ids"]) for x in encoded_inputs]
                padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
                attention_mask = (padded != 0).long()
                return {"input_ids": padded, "attention_mask": attention_mask}

        tokenizer = MockTokenizer()
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3], "advantages": 0.5},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5], "advantages": 1.0},
        ]

        collator = TPODataCollator(tokenizer=tokenizer)
        batch = collator(features)
        assert "advantages" in batch
        assert batch["advantages"].shape[0] == 2
        assert batch["advantages"][0].item() == pytest.approx(0.5)

    def test_collator_handles_list_advantages(self):
        from tpo_torch.trainer import TPODataCollator

        class MockTokenizer:
            pad_token_id = 0
            pad_token = "<pad>"

            def pad(self, encoded_inputs, padding=True, max_length=None,
                    pad_to_multiple_of=None, return_tensors="pt"):
                from torch.nn.utils.rnn import pad_sequence
                input_ids = [torch.tensor(x["input_ids"]) for x in encoded_inputs]
                padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
                attention_mask = (padded != 0).long()
                return {"input_ids": padded, "attention_mask": attention_mask}

        tokenizer = MockTokenizer()
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "advantages": [0.1, 0.2, 0.3]},
            {"input_ids": [4, 5, 6], "labels": [4, 5, 6], "advantages": [0.4, 0.5, 0.6]},
        ]
        collator = TPODataCollator(tokenizer=tokenizer)
        batch = collator(features)
        assert batch["advantages"].shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
