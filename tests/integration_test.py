import torch
import torch.nn as nn
from transformers import TrainingArguments, PretrainedConfig, PreTrainedModel
from datasets import Dataset
import numpy as np
from tpo_torch.trainer import TPOTrainer


class TinyConfig(PretrainedConfig):
    model_type = "tiny"

    def __init__(self, vocab_size: int = 100, hidden_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        logits = self.output(x)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)

def main():
    print("[*] Starting LITE Integration Test...")
    config = TinyConfig()
    model = TinyModel(config)
    ref_model = TinyModel(config)
    
    # Create tiny dataset
    data = []
    for _ in range(10):
        data.append({
            "input_ids": torch.randint(0, 100, (8,)).tolist(),
            "attention_mask": [1]*8,
            "labels": torch.randint(0, 100, (8,)).tolist(),
            "advantages": float(np.random.uniform(0.1, 1.0))
        })
    dataset = Dataset.from_list(data)
    
    args = TrainingArguments(
        output_dir="./tiny_test",
        max_steps=5,
        per_device_train_batch_size=2,
        logging_steps=1,
        remove_unused_columns=False
    )
    
    trainer = TPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset,
    )
    
    print("[*] Running 5 steps of TPO on Tiny model...")
    trainer.train()
    print("[*] Integration Test Successful!")

if __name__ == "__main__":
    main()
