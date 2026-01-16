from transformers import Qwen2Config

class ModelConfig(Qwen2Config):
    def __init__(
            self,
            base_model_name_or_path="Qwen2.5-0.5B",
            model_max_length=8192,
            spk_emb_dim=256,
            spk_aware=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.model_max_length = model_max_length
        self.spk_emb_dim = spk_emb_dim
        self.spk_aware = spk_aware