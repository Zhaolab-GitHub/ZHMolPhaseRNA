from transformers import PretrainedConfig


class ErnieRNAConfig(PretrainedConfig):
    model_type = "ernie_rna"

    auto_map = {
        "AutoConfig": "configuration_ernierna.ErnieRNAConfig",
        "AutoModel": "modeling_ernierna.ErnieRNAModel",
        "AutoModelForMaskedLM": "modeling_ernierna.ErnieRNAForMaskedLM",
    }

    def __init__(
        self,
        vocab_size=25,
        num_layers=12,
        embed_dim=768,
        ffn_embed_dim=3072,
        attention_heads=12,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        activation_fn="gelu",
        max_positions=1024,
        padding_idx=1,
        mask_idx=24,
        num_segments=2,
        model_max_length=1024,
        **kwargs,
    ):
        super().__init__(padding_idx=padding_idx, **kwargs)
        self.auto_map = dict(type(self).auto_map)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation_fn
        self.max_positions = max_positions
        self.mask_idx = mask_idx
        self.num_segments = num_segments
        self.model_max_length = model_max_length
