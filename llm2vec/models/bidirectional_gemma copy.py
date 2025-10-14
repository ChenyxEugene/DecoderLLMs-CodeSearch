import torch
from typing import List, Optional, Tuple, Union


from transformers import GemmaModel, GemmaForCausalLM, GemmaPreTrainedModel, GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaDecoderLayer,
    GemmaAttention,
    GemmaFlashAttention2,
    GemmaSdpaAttention,
    GemmaMLP,
    GemmaRMSNorm,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

from torch import nn
from transformers.utils import logging
from peft import PeftModel

from .attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)



logger = logging.get_logger(__name__)

class ModifiedGemmaAttention(GemmaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

class ModifiedGemmaFlashAttention2(GemmaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

class ModifiedGemmaSdpaAttention(GemmaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

GEMMA_ATTENTION_CLASSES = {
    "eager": ModifiedGemmaAttention,
    "flash_attention_2": ModifiedGemmaFlashAttention2,
    "sdpa": ModifiedGemmaSdpaAttention,
}
# # from transformers.models.gemma.modeling_gemma 中gemma_attention_类定义
# GEMMA_ATTENTION_CLASSES = {
#     "eager": GemmaAttention,
#     "flash_attention_2": GemmaFlashAttention2,
#     "sdpa": GemmaSdpaAttention,
# }


class ModifiedGemmaDecoderLayer(GemmaDecoderLayer):
    def __init__(self, config: GemmaConfig, layer_idx: int):
    # def __init__(self, config: GemmaConfig, layer_idx: int):

        nn.Module.__init__(self) ## 只加入了这个用于初始化 Initially, super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GemmaBiModel(GemmaModel):
    def __init__(self, config: GemmaConfig):
        GemmaPreTrainedModel.__init__(self, config) # Initially, super().__init__(config) 重新初始化super()初始化方法

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()




class GemmaBiForMNTP(GemmaForCausalLM):
    def __init__(self, config):
        GemmaPreTrainedModel.__init__(self, config) # Initially, super().__init__(config)
        
        self.model = GemmaBiModel(config)  # Initially, LlamaModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()


    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.model.save_pretrained(path)
