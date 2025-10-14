# starcoder2_bi_model.py

import torch
from torch import nn
from typing import List, Optional, Tuple, Union

from peft import PeftModel

from transformers import Starcoder2Config, Starcoder2Model, Starcoder2ForCausalLM, Starcoder2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2DecoderLayer,
    Starcoder2Attention,
    Starcoder2FlashAttention2,
    Starcoder2SdpaAttention,
    Starcoder2MLP,
    _prepare_4d_causal_attention_mask, # 仍然需要导入，但我们会在forward中绕过它
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

# --- 步骤 1: 创建修改后的注意力类 ---
# 核心修改：在初始化时，将 is_causal 强制设为 False。
# 这会告知 flash_attention 和 sdpa 实现不要使用因果掩码。

class ModifiedStarcoder2Attention(Starcoder2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedStarcoder2FlashAttention2(Starcoder2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedStarcoder2SdpaAttention(Starcoder2SdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

# 创建一个字典来映射不同的注意力实现到我们修改后的类
MODIFIED_STARCODER2_ATTENTION_CLASSES = {
    "eager": ModifiedStarcoder2Attention,
    "flash_attention_2": ModifiedStarcoder2FlashAttention2,
    "sdpa": ModifiedStarcoder2SdpaAttention,
}


# --- 步骤 2: 创建修改后的解码器层 ---
# 这个类将使用上面定义的、修改过的非因果注意力模块。

class ModifiedStarcoder2DecoderLayer(Starcoder2DecoderLayer):
    def __init__(self, config: Starcoder2Config, layer_idx: int):
        # nn.Module.__init__(self) # Starcoder2DecoderLayer的父类已经是nn.Module，不需要重新调用
        super(Starcoder2DecoderLayer, self).__init__() # 使用super()来正确初始化父类
        self.hidden_size = config.hidden_size

        # 使用我们自己的注意力类字典来实例化注意力模块
        self.self_attn = MODIFIED_STARCODER2_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Starcoder2MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)


# --- 步骤 3: 创建双向模型基座 ---
# 这个类继承自 Starcoder2Model，但使用我们修改后的解码器层，
# 并且最重要的是，它重写了 forward 方法来改变掩码（mask）的生成方式。

class Starcoder2BiModel(Starcoder2Model):
    _no_split_modules = ["ModifiedStarcoder2DecoderLayer"]

    def __init__(self, config: Starcoder2Config):
        # 调用 Starcoder2PreTrainedModel 的 __init__
        Starcoder2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embedding_dropout = config.embedding_dropout
        
        # 使用我们修改后的解码器层
        self.layers = nn.ModuleList(
            [ModifiedStarcoder2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 这个 forward 方法的大部分内容是从原始 Starcoder2Model.forward 复制而来
        # 关键的区别在于 attention_mask 的处理方式
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            # past_key_values a Cache object
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # ========================== 关键修改点 ==========================
        # 原始代码会调用 _prepare_4d_causal_attention_mask 等函数来强制创建一个因果掩码。
        # 我们要绕过这个逻辑，只根据 padding 创建一个双向的掩码。
        
        if self._attn_implementation == "flash_attention_2":
            # Flash Attention 2可以直接处理2D的padding mask，并且会根据我们修改的 is_causal=False 来决定是否应用因果关系。
            # 我们只需要确保在有padding时传递mask即可。
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 对于 'eager' 和 'sdpa' 实现，我们需要一个4D的注意力掩码。
            # 我们从2D的padding mask手动创建一个4D的“双向”掩码。
            if attention_mask is not None:
                # 从 (batch_size, seq_length) 创建 (batch_size, 1, 1, seq_length)
                # 这允许每个token关注所有未被padding的token
                expanded_mask = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
                # 将 1 (有效) 变为 0.0, 0 (填充) 变为一个很大的负数。
                # 这是注意力中标准的additive mask。
                attention_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
            # 如果 attention_mask 原本就是 None, 那就保持 None, 意味着没有 token 被 mask.
        # ======================= 修改结束 ============================

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.embedding_dropout, training=self.training)

        # Gradient checkpointing
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# --- 步骤 4: 创建最终的 MNTP 模型 ---
# 这个顶层类提供了与 `...ForCausalLM` 类似的接口，
# 并且包含了用于 PEFT 训练的辅助函数。

class Starcoder2BiForMNTP(Starcoder2ForCausalLM):
    def __init__(self, config):
        # 调用 Starcoder2PreTrainedModel 的 __init__
        Starcoder2PreTrainedModel.__init__(self, config)
        # 使用我们自己的双向模型 Starcoder2BiModel
        self.model = Starcoder2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    # 为 PEFT 模型添加 getter
    def get_model_for_peft(self):
        return self.model

    # 为 PEFT 模型添加 setter
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # 保存 PEFT 模型
    def save_peft_model(self, path):
        self.model.save_pretrained(path)