import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import PretrainedConfig, PreTrainedModel, TrainingArguments
from datasets import Dataset


class TinyConfig(PretrainedConfig):
    model_type = "tiny"

    def __init__(self, vocab_size=64, hidden_size=16, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        from types import SimpleNamespace

        x = self.embed(input_ids)
        logits = self.head(x)
        return SimpleNamespace(logits=logits)


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<pad>"

    def pad(self, encoded_inputs, padding=True, max_length=None,
            pad_to_multiple_of=None, return_tensors="pt"):
        from torch.nn.utils.rnn import pad_sequence
        input_ids = [torch.tensor(x["input_ids"]) for x in encoded_inputs]
        padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = (padded != 0).long()
        return {"input_ids": padded, "attention_mask": attention_mask}

    def save_pretrained(self, *args, **kwargs):
        pass

    def save_vocabulary(self, *args, **kwargs):
        pass


class TestTPOEndToEnd:
    def test_trainer_uses_correct_loss(self):
        from tpo_torch.trainer import TPOTrainer

        config = TinyConfig()
        model = TinyModel(config)
        ref_model = TinyModel(config)
        tokenizer = MockTokenizer()

        data = []
        for i in range(4):
            data.append({
                "input_ids": [i + 1, i + 2, i + 3],
                "labels": [i + 1, i + 2, i + 3],
                "advantages": float(i % 2) + 0.1,
            })
        train_dataset = Dataset.from_list(data)

        args = TrainingArguments(
            output_dir="./test_output",
            max_steps=2,
            per_device_train_batch_size=2,
            logging_steps=1,
            save_steps=999,
            learning_rate=2e-5,
            remove_unused_columns=False,
            report_to=["none"],
        )

        trainer = TPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=0.1,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        losses = []
        original_compute = trainer.compute_loss

        def capturing_compute_loss(mdl, inp, ret=False, num_items_in_batch=None):
            loss, _ = original_compute(
                mdl, inp, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            losses.append(loss.item())
            return loss

        trainer.compute_loss = capturing_compute_loss
        trainer.train()

        assert len(losses) >= 1, "Trainer should have computed at least 1 loss"
        assert all(not (l != l) for l in losses), "No NaN losses"
        assert all(l >= -1e-6 for l in losses), f"All losses >= ~0, got {losses}"

    def test_no_ref_model_uses_policy_as_ref(self):
        from tpo_torch.trainer import TPOTrainer

        config = TinyConfig()
        model = TinyModel(config)
        tokenizer = MockTokenizer()

        data = [
            {"input_ids": [1, 2], "labels": [1, 2], "advantages": 0.5},
            {"input_ids": [3, 4], "labels": [3, 4], "advantages": 0.5},
        ]
        train_dataset = Dataset.from_list(data)

        args = TrainingArguments(
            output_dir="./test_output",
            max_steps=1,
            per_device_train_batch_size=2,
            save_steps=999,
            remove_unused_columns=False,
            report_to=["none"],
        )

        trainer = TPOTrainer(
            model=model,
            ref_model=None,
            beta=0.1,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )

        assert trainer.ref_model is None

        losses = []
        original_compute = trainer.compute_loss

        def capture_loss(mdl, inp, ret=False, num_items_in_batch=None):
            loss, _ = original_compute(
                mdl, inp, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            losses.append(loss.item())
            return loss

        trainer.compute_loss = capture_loss
        trainer.train()

        assert len(losses) == 1
        assert not (losses[0] != losses[0]), "Loss should not be NaN"
        assert losses[0] >= -1e-6, f"Loss >= ~0, got {losses[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
