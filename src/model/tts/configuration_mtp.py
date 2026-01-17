from transformers import (
    Qwen2Config,
)



class MedusaConfig(Qwen2Config):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 3.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
            self,
            medusa_num_heads=3,
            medusa_num_layers=1,
            model_max_length=8192,
            base_model_name_or_path="Qwen2.5-0.5B",  # need to be force overwrite to avoid unexpected network request
            head_implement="linear",
            head_transformers_layer=2,
            spk_aware=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_max_length = model_max_length
        self.medusa_num_heads = medusa_num_heads
        self.base_model_name_or_path = base_model_name_or_path
        self.head_implement = head_implement
        self.head_transformers_layer = head_transformers_layer
        self.spk_aware = spk_aware