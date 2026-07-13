"""TPO-Torch Trainer — HuggingFace Trainer subclass with TPO loss."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Any, Optional, Union

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .loss import tpo_loss_from_logits


@dataclass
class TPODataCollator:
    """Data collator that preserves the 'advantages' column through batching.

    HuggingFace's default collator drops columns that are not model inputs.
    This collator keeps 'advantages' intact so the TPO loss can use them.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}

        input_ids = [f["input_ids"] for f in features]
        encoded = self.tokenizer.pad(
            [{"input_ids": ids} for ids in input_ids],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["input_ids"] = encoded["input_ids"]
        batch["attention_mask"] = encoded["attention_mask"]

        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            pad_length = batch["input_ids"].size(1)
            padded_labels = []
            for lbl in labels:
                if len(lbl) < pad_length:
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
                    lbl = lbl + [pad_id] * (pad_length - len(lbl))
                padded_labels.append(lbl[:pad_length])
            batch["labels"] = torch.tensor(padded_labels)

        advantages = [f["advantages"] for f in features]
        if isinstance(advantages[0], float):
            batch["advantages"] = torch.tensor(advantages)
        elif isinstance(advantages[0], list):
            max_adv_len = max(len(a) for a in advantages)
            padded = [a + [0.0] * (max_adv_len - len(a)) for a in advantages]
            batch["advantages"] = torch.tensor(padded)
        elif isinstance(advantages[0], torch.Tensor):
            batch["advantages"] = torch.stack(advantages)
        else:
            raise ValueError(f"Unsupported advantages dtype: {type(advantages[0])}")

        return batch


class TPOTrainer(Trainer):
    """Target Policy Optimization Trainer.

    Subclasses HuggingFace's ``Trainer`` to use the TPO loss function.
    Expects a dataset with an ``advantages`` column (float: higher = better).

    Args:
        model: Policy model to train.
        ref_model: Frozen reference model. If ``None``, the policy model is
            used as its own reference (no KL anchor).
        beta: Temperature parameter controlling target distribution sharpness.
        args: HuggingFace ``TrainingArguments``.
        train_dataset: Training dataset with ``advantages`` column.
        eval_dataset: Optional evaluation dataset.
        processing_class: Tokenizer or processor.
        max_seq_length: Maximum sequence length for padding.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        ref_model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None,
        beta: float = 0.1,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        processing_class: Optional[Any] = None,
        max_seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if processing_class is None and "tokenizer" in kwargs:
            processing_class = kwargs.pop("tokenizer")

        if processing_class is not None and "data_collator" not in kwargs:
            kwargs["data_collator"] = TPODataCollator(
                tokenizer=processing_class,
                max_length=max_seq_length or 512,
            )

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs,
        )
        self.ref_model = ref_model
        self.beta = beta

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            ref_device = next(ref_model.parameters()).device
            model_device = next(model.parameters()).device
            if ref_device != model_device:
                self.ref_model = self.ref_model.to(model_device)

    def _get_train_dataloader(self):
        """Override to preserve 'advantages' column even with remove_unused_columns."""
        dataloader = super()._get_train_dataloader()
        original_collate = dataloader.collate_fn

        def wrapping_collator(batch):
            result = original_collate(batch)
            if isinstance(result, dict) and "advantages" not in result:
                for feature in batch:
                    if "advantages" in feature:
                        advs = [f["advantages"] for f in batch]
                        if isinstance(advs[0], float):
                            result["advantages"] = torch.tensor(advs)
                        elif isinstance(advs[0], list):
                            max_len = max(len(a) for a in advs)
                            padded = [a + [0.0] * (max_len - len(a)) for a in advs]
                            result["advantages"] = torch.tensor(padded)
                        elif isinstance(advs[0], torch.Tensor):
                            result["advantages"] = torch.stack(advs)
                        break
            return result

        dataloader.collate_fn = wrapping_collator
        return dataloader

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute TPO loss for a batch."""
        advantages = inputs.get("advantages")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if advantages is None:
            raise ValueError(
                "TPOTrainer requires 'advantages' in the inputs batch. "
                "Ensure your dataset contains an 'advantages' column."
            )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        policy_logits = outputs.logits

        labels = inputs.get("labels", input_ids)

        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
        else:
            ref_logits = policy_logits.detach()

        loss = tpo_loss_from_logits(
            policy_logits=policy_logits,
            reference_logits=ref_logits,
            labels=labels,
            advantages=advantages,
            beta=self.beta,
            attention_mask=attention_mask,
        )

        return (loss, outputs) if return_outputs else loss
