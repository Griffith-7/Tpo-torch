from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


class TPOModel(PreTrainedModel):
    """
    Base model wrapper for TPO training.
    Provides a frozen reference-policy mechanism via a companion ref_model.
    """

    load_weight_prefix = "tpo_ref"

    def __init__(self, config: AutoConfig, ref_model_name: str = None, **kwargs):
        super().__init__(config)
        self.ref_model = None
        self._ref_frozen = False

        if ref_model_name is not None:
            self._init_reference(ref_model_name)

    def _init_reference(self, model_name: str):
        """Initialize a frozen copy of the policy as the reference model."""
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.freeze_reference_policy()

    def freeze_reference_policy(self):
        """Freeze the reference model so its weights never change during training."""
        if self.ref_model is None:
            return
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        self._ref_frozen = True

    def unfreeze_reference_policy(self):
        """Unfreeze the reference model for continued fine-tuning."""
        if self.ref_model is None:
            return
        for param in self.ref_model.parameters():
            param.requires_grad = True
        self.ref_model.train()
        self._ref_frozen = False

    @property
    def is_reference_frozen(self) -> bool:
        return self._ref_frozen
