import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

try:
    from .configuration_ernierna import ErnieRNAConfig
except ImportError:
    from configuration_ernierna import ErnieRNAConfig


class ErnieRNASinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions, embed_dim, padding_idx):
        super().__init__()
        self.embedding_dim = embed_dim
        self.padding_idx = padding_idx
        # Table size: need indices up to padding_idx + 1 + num_positions
        table_size = padding_idx + 1 + num_positions
        self.register_buffer("weights", self._get_embedding(table_size, embed_dim, padding_idx))

    @staticmethod
    def _get_embedding(num_embeddings, embedding_dim, padding_idx):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input_ids):
        mask = input_ids.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1) * mask).long() + self.padding_idx
        return self.weights.index_select(0, positions.view(-1)).view(
            input_ids.shape[0], input_ids.shape[1], -1
        ).detach()


class ErnieRNATwodProj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(1, 6)
        self.linear2 = nn.Linear(6, config.attention_heads)
        self.activation_fn = ACT2FN[config.activation_fn]

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


def _compute_pairing_bias(input_ids):
    B, T = input_ids.shape
    xi = input_ids.unsqueeze(2).expand(B, T, T)
    xj = input_ids.unsqueeze(1).expand(B, T, T)

    score = torch.zeros(B, T, T, dtype=torch.float32, device=input_ids.device)
    score[(xi == 5) & (xj == 6)] = 2.0
    score[(xi == 6) & (xj == 5)] = 2.0
    score[(xi == 4) & (xj == 7)] = 3.0
    score[(xi == 7) & (xj == 4)] = 3.0
    score[(xi == 4) & (xj == 6)] = 0.8
    score[(xi == 6) & (xj == 4)] = 0.8
    return score.unsqueeze(-1)  # [B, T, T, 1]


class ErnieRNAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def _to_bh_t_hd(self, tensor, tgt_len, bsz):
        return tensor.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, x, key_padding_mask=None, twod_bias=None, output_attentions=False):
        tgt_len, bsz, _ = x.size()

        q = self._to_bh_t_hd(self.q_proj(x), tgt_len, bsz)
        k = self._to_bh_t_hd(self.k_proj(x), tgt_len, bsz)
        v = self._to_bh_t_hd(self.v_proj(x), tgt_len, bsz)

        scale = self.head_dim ** -0.5
        q = q * scale

        attn_weights = torch.bmm(q, k.transpose(-2, -1))  # [B*H, T, T]

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, tgt_len)

        if twod_bias is not None:
            attn_weights = attn_weights + twod_bias.reshape(bsz * self.num_heads, tgt_len, tgt_len)

        # Pre-softmax attention becomes the 2D bias for the next layer
        twod_bias_new = attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_weights_out = None
        if output_attentions:
            attn_weights_out = attn_probs.view(bsz, self.num_heads, tgt_len, tgt_len)
        attn_probs = self.dropout(attn_probs)

        out = torch.bmm(attn_probs, v)
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        out = self.out_proj(out)

        return out, attn_weights_out, twod_bias_new


class ErnieRNALayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ErnieRNAAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.embed_dim)
        self.fc1 = nn.Linear(config.embed_dim, config.ffn_embed_dim)
        self.fc2 = nn.Linear(config.ffn_embed_dim, config.embed_dim)
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.activation_fn = ACT2FN[config.activation_fn]

    def forward(self, x, key_padding_mask=None, twod_bias=None, output_attentions=False):
        residual = x
        x, attn_weights, twod_bias_new = self.self_attn(
            x,
            key_padding_mask=key_padding_mask,
            twod_bias=twod_bias,
            output_attentions=output_attentions,
        )
        x = self.dropout(x)
        x = self.self_attn_layer_norm(residual + x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.final_layer_norm(residual + x)

        return x, attn_weights, twod_bias_new


class ErnieRNAModel(PreTrainedModel):
    config_class = ErnieRNAConfig
    base_model_prefix = "model"
    _supports_sdpa = False
    _supports_flash_attn_2 = False

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.padding_idx

        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.padding_idx)
        self.embed_positions = ErnieRNASinusoidalPositionalEmbedding(
            config.max_positions, config.embed_dim, config.padding_idx
        )
        self.segment_embeddings = nn.Embedding(config.num_segments, config.embed_dim)
        self.emb_layer_norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ErnieRNALayer(config) for _ in range(config.num_layers)])
        self.twod_proj = ErnieRNATwodProj(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # HF: 1=attend, 0=pad -> True=padding
        if attention_mask is not None:
            padding_mask = attention_mask.eq(0)
        else:
            padding_mask = input_ids.eq(self.padding_idx)

        # Zero out padding positions after masking (matches fairseq behavior)
        x = self.embed_tokens(input_ids)
        # Sinusoidal PE is a float32 buffer; cast to activation dtype for bfloat16 compat.
        x = x + self.embed_positions(input_ids).to(x.dtype)
        if token_type_ids is not None:
            x = x + self.segment_embeddings(token_type_ids)
        x = self.emb_layer_norm(x)
        if padding_mask.any():
            x = x * (~padding_mask).unsqueeze(-1).to(x.dtype)
        x = self.dropout(x)

        # Compute initial 2D bias from sequence (always float32 as in original)
        pairing = _compute_pairing_bias(input_ids)  # [B, T, T, 1]
        twod_proj_f32 = self.twod_proj.float()
        twod_bias = twod_proj_f32(pairing.float())  # [B, T, T, H]
        twod_bias = twod_bias.permute(0, 3, 1, 2).contiguous().to(x.dtype)  # [B, H, T, T]

        # Transpose to [T, B, C] for attention
        x = x.transpose(0, 1)

        all_hidden_states = []
        all_attentions = []

        if output_hidden_states:
            all_hidden_states.append(x.transpose(0, 1))

        key_padding_mask = padding_mask if padding_mask.any() else None

        for layer in self.layers:
            x, attn_weights, twod_bias = layer(
                x,
                key_padding_mask=key_padding_mask,
                twod_bias=twod_bias,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                all_hidden_states.append(x.transpose(0, 1))
            if output_attentions:
                all_attentions.append(attn_weights)

        x = x.transpose(0, 1)  # [B, T, C]

        if not return_dict:
            return tuple(v for v in [x, tuple(all_hidden_states) or None, tuple(all_attentions) or None] if v is not None)

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
        )


class ErnieRNALMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.decoder = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        x = self.layer_norm(self.activation_fn(self.dense(x)))
        x = self.decoder(x)
        return x


class ErnieRNAForMaskedLM(PreTrainedModel):
    config_class = ErnieRNAConfig
    base_model_prefix = "model"
    _supports_sdpa = False
    _supports_flash_attn_2 = False

    def __init__(self, config):
        super().__init__(config)
        self.model = ErnieRNAModel(config)
        self.lm_head = ErnieRNALMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(out[0])

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,) + out[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )
