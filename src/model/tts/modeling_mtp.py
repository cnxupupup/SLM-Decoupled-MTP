import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    Qwen2PreTrainedModel,
    AutoTokenizer
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from safetensors.torch import load_file
from torch.nn import CrossEntropyLoss
import os
import glob
from torch.nn import functional as F
from transformers.trainer_pt_utils import LabelSmoother
from .configuration_mtp import MedusaConfig

IGNORE_TOKEN_ID = LabelSmoother.ignore_index




class SubLlmHead(nn.Module):
    def __init__(self, config, mapping, name="a", implement="linear"):
        # mapping is a tensor of shape (vocab_size,)
        super().__init__()
        self.name = name
        self.hidden_size = config.hidden_size
        self.implement = implement

        if not isinstance(mapping, torch.Tensor):
            mapping = torch.tensor(mapping)

        bias = torch.min(mapping)
        orig_to_small = torch.full((torch.max(mapping) - bias + 2,), -100)
        keys = torch.arange(0, mapping.shape[-1])
        reverse_mapping = orig_to_small.scatter(0, mapping - bias, keys)

        self.register_buffer("bias", bias)
        self.register_buffer("small_to_orig", mapping)
        self.register_buffer("orig_to_small", reverse_mapping)

        self.vocab_size = mapping.shape[-1]

        # mlp
        if implement == "linear":
            self.head = nn.Linear(self.hidden_size, self.vocab_size)
        elif implement == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, 2 * self.hidden_size),
                nn.SiLU(),
                nn.Linear(2 * self.hidden_size, self.vocab_size)
            )
        elif implement == "transformer":
            self.head_layers = nn.ModuleList(
                [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.head_transformers_layer)]
            )
            self.head_norm = nn.LayerNorm(self.hidden_size, config.rms_norm_eps)
            self.head_proj = nn.Linear(self.hidden_size, self.vocab_size)
        else:
            raise ValueError(f"supported implement: {implement}")

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None
    ):
        # [b,n,hidden_size] -> [b,n,vocab_size]
        if self.implement in ["linear", "mlp"]:
            return self.head(hidden_states)
        else:
            for decoder_layer in self.head_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            return self.head_proj(self.head_norm(hidden_states))

    def mapping_sub_to_orig(self, input_ids, ignore_idx=-100):
        # small to large
        # input_ids is a tensor of shape (batch_size, seq_len)
        ignore_mask = input_ids == ignore_idx
        input_ids[ignore_mask] = 0
        return torch.where(ignore_mask, ignore_idx, self.small_to_orig[input_ids])

    def mapping_orig_to_sub(self, input_ids):
        # large to small
        # inputs_ids is a tensor of shape (batch_size, seq_len)
        lookup_idx = input_ids - self.bias
        lookup_idx = torch.where(lookup_idx >= 0, lookup_idx, self.orig_to_small.shape[-1] - 1)
        return self.orig_to_small[lookup_idx]


class MedusaPreTrainedModelForQwen2(Qwen2PreTrainedModel):
    config_class = MedusaConfig


class MedusaModelForQwen2Base(MedusaPreTrainedModelForQwen2):
    """The Medusa Language Model compatible with Qwen2(2.5) series.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
            self,
            config: MedusaConfig,
    ):
        """
        Args:
            config: MedusaConfig
        """
        super().__init__(config)
        self.config = config
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=config.torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            model_max_length=config.model_max_length,
            padding_side="right",
            use_fast=False,
            bos_token="<|im_start|>"
        )

        self._add_tokens()
        self.hidden_size = config.hidden_size
        self.placeholder_index = self.tokenizer.convert_tokens_to_ids("<|placeholder|>")

        if self.config.spk_aware:
            self.spk_emb_proj = nn.Sequential(
            nn.Linear(config.spk_emb_dim, 4 * config.spk_emb_dim, dtype=self.lm_model.dtype),
            nn.SiLU(),
            nn.Linear(4 * config.spk_emb_dim, config.hidden_size, dtype=self.lm_model.dtype)
        )
        else:
            self.spk_emb_proj = None

        self.hidden_size = config.hidden_size
        self.medusa = config.medusa_num_heads
        self.vocab_size = config.vocab_size = len(self.tokenizer)

        # Create a list of Medusa heads
        if self.medusa == 3:
            self.speech_heads = nn.ModuleList(
                [
                    SubLlmHead(
                        config=self.config,
                        mapping=self.tokenizer.convert_tokens_to_ids(
                            [f"<{name[0]}{i}>" for i in range(1024)] + ["<eosp>"] + (
                                [] if name in ["a1", "b1", "c1"] else ["<|speech_pad|>"])),
                        name=name,
                        implement=config.head_implement
                    )
                    for name in ["a1", "b1", "c1"]
                ]
            )
        elif self.medusa == 6:
            self.speech_heads = nn.ModuleList(
                [
                    SubLlmHead(
                        config=self.config,
                        mapping=self.tokenizer.convert_tokens_to_ids(
                            [f"<{name[0]}{i}>" for i in range(1024)] + ["<eosp>"] + (
                                [] if name in ["a1", "b1", "c1"] else ["<|speech_pad|>"])),
                        name=name,
                        implement=config.head_implement
                    )
                    for name in ["a1", "b1", "c1", "a2", "b2", "c2"]
                ]
            )
        elif self.medusa == 12:
            self.speech_heads = nn.ModuleList(
                [
                    SubLlmHead(
                        config=self.config,
                        mapping=self.tokenizer.convert_tokens_to_ids(
                            [f"<{name[0]}{i}>" for i in range(1024)] + ["<eosp>"] + (
                                [] if name in ["a1", "b1", "c1"] else ["<|speech_pad|>"])),
                        name=name,
                        implement=config.head_implement
                    )
                    for name in ["a1", "b1", "c1", "a2", "b2", "c2", "a3", "b3", "c3", "a4", "b4", "c4"]
                ]
            )
        else:
            raise NotImplementedError(f"{self.medusa} heads not supported for now !")

        # Ensure medusa_head's dtype and device align with the base_model
        for head in self.speech_heads:
            head.to(self.lm_model.dtype).to(self.lm_model.device)

        if self.training:
            self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            **kwargs,
    ):
        config = MedusaConfig.from_pretrained(model_name_or_path)
        model = cls(config, **kwargs)
        possible_path = os.path.join(model_name_or_path, "model.safetensors")
        if os.path.exists(possible_path):
            state_dict = load_file(possible_path)
            if 'lm_model.model.embed_tokens.weight' in state_dict:
                state_dict["lm_model.lm_head.weight"] = state_dict['lm_model.model.embed_tokens.weight']
            model.load_state_dict(state_dict)
        else:
            state_dict = {}
            paths = glob.glob(os.path.join(model_name_or_path, '*.safetensors'))
            for path in paths:
                state_part = load_file(path)
                state_dict.update(state_part)
            if 'lm_model.model.embed_tokens.weight' in state_dict:
                state_dict["lm_model.lm_head.weight"] = state_dict['lm_model.model.embed_tokens.weight']
            model.load_state_dict(state_dict)
        return model

    def get_sosp_eosp_idx(self):
        return self.tokenizer.convert_tokens_to_ids(['<sosp>', '<eosp>'])

    def get_pad_idx(self):
        return self.tokenizer.convert_tokens_to_ids(["<|speech_pad|>"])

    def _add_tokens(self):
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        prefix_special_tokens, suffix_special_tokens = ['[Human]', '[Assistant]', '<sosp>', '<|placeholder|>'], ['<eosp>',
                                                                                               "<|speech_pad|>"]

        if "<sosp>" not in self.tokenizer.get_vocab():
            units_size = 1024
            new_tokens = prefix_special_tokens + [f"<a{x}>" for x in range(units_size)] + [f"<b{x}>" for x in
                                                                                           range(units_size)] + [
                             f"<c{x}>" for x in range(units_size)] + suffix_special_tokens
            self.tokenizer.add_tokens(new_tokens)
        for token in prefix_special_tokens:
            if token not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([token])
        if '<eosp>' not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(['<eosp>'])

        # resize embedding
        embedding_size = self.lm_model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.lm_model.resize_token_embeddings(len(self.tokenizer))

    def inference_forward(
            self,
            inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None,
            text=False
    ):
        outputs = self.lm_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )
        if text:
            text_logits = self.lm_model.lm_head(outputs[0])
            return text_logits, outputs

        hidden_states = outputs[0]

        speech_logits_1 = []
        for layer in self.speech_heads[:3]:
            speech_logits_layer = layer(hidden_states)
            speech_logits_1.append(speech_logits_layer)

        speech_logits_2 = []
        for layer in self.speech_heads[3:]:
            speech_logits_layer = layer(hidden_states)
            speech_logits_2.append(speech_logits_layer)

        return torch.stack(speech_logits_1, dim=0), None, outputs
    

    def compute_speech_loss(self, logits, labels, speech_mask, speech_head):
        logits_shifted = logits[:, :-1]
        labels_mask = torch.where(speech_mask.bool(), labels, -100)
        labels_mask = speech_head.mapping_orig_to_sub(labels_mask)
        labels_mask_shifted = labels_mask[:, 1:]
        head_loss = self.loss_fct(logits_shifted.contiguous().view(-1, logits_shifted.shape[-1]),
                                  labels_mask_shifted.contiguous().view(-1))
        _, topk = logits_shifted.topk(1, dim=-1)
        top_one = topk.eq(labels_mask_shifted.unsqueeze(-1)).any(-1)
        top_one_corr = top_one[speech_mask[:, 1:].bool()].float().mean().item()
        return head_loss, top_one_corr

    def get_nucleus_one_token(self, logit, temperature, top_p=0.8, top_k=50):

        logit = logit / temperature

        if top_k > 0:
            top_k = min(top_k, logit.shape[-1])
            top_k_logits, top_k_indices = torch.topk(logit, top_k, dim=-1)
            mask = torch.ones_like(logit) * float("-inf")
            mask.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
            logit = mask

        if top_p < 1.0:
            probs = torch.softmax(logit, dim=-1)
            sorted_logits, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_logits, dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices,
                                                                 src=sorted_indices_to_remove)
            logit[indices_to_remove] = float('-inf')

        sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
        return sampled_tokens

    def get_greedy_token(self, logits):
        _, sampled_tokens = torch.max(F.softmax(logits, dim=-1), dim=-1)
        return sampled_tokens.unsqueeze(1)

    @torch.inference_mode()
    def generate(
            self,
            input_ids,
            spk_emb=None,
            attention_mask=None,
            use_cache=True,
            temperature=1.0,
            repetition_penalty=1.5,
            do_sample=False,
            top_p=0.8,
            top_k=50,
            max_steps=512,
    ):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now !!"
        past_key_values = None
        history_speech_tokens = []
        # history_speech_tokens = torch.zeros(input_ids.shape[0], self.medusa, 0).to(input_ids.device)
        for _ in range(len(self.speech_heads)):
            history_speech_tokens.append(torch.zeros(input_ids.shape[0], 0).to(input_ids.device))

        input_len = input_ids.shape[1]
        assert input_len > 0, "input length must > 0"

        new_token = 0
        sosp_idx, eosp_idx = self.get_sosp_eosp_idx()
        pad_idx = self.get_pad_idx()

        token_embed = self.lm_model.get_input_embeddings()

        placeholder_mask = (input_ids == self.placeholder_index)
        if self.spk_emb_proj:
            spk_emb = self.spk_emb_proj(spk_emb)    # proj
            spk_emb = spk_emb.unsqueeze(0)
            spk_emb_expanded = spk_emb.expand(-1, input_ids.shape[1], -1)   # expand
            input_embs = token_embed(input_ids)
            input_embs = torch.where(
                placeholder_mask.unsqueeze(-1).expand(-1, -1, input_embs.size(-1)).bool(),
                spk_emb_expanded,
                input_embs
            )
        else:
            input_embs = token_embed(input_ids)

        inference_mode = "text"


        for _ in range(max_steps):
            if inference_mode == "text":
                text_logits, outputs = self.inference_forward(
                    # input_embeds
                    input_embeds if past_key_values is None else input_embeds[:, -1:],
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    text=True
                )
                last_logits = text_logits[:, -1]
                if not do_sample:
                    sampled_tokens = self.get_greedy_token(
                        last_logits
                    )
                else:
                    sampled_tokens = self.get_nucleus_one_token(
                        last_logits,
                        temperature=temperature,
                        top_p=top_p
                    )
                new_token += sampled_tokens.shape[-1]
                input_ids = torch.cat([input_ids, sampled_tokens], dim=1).to(torch.long)
                input_embeds = torch.cat([input_embeds, token_embed(sampled_tokens)], dim=1)

                if sampled_tokens[0, -1] == sosp_idx:  # change inference mode
                    inference_mode = "speech"

            elif inference_mode == "speech":
                speech_logits_1, speech_logits_2, outputs = self.inference_forward(
                    input_embeds if past_key_values is None else input_embeds[:, -1:],
                    use_cache=use_cache,
                    past_key_values=past_key_values
                )
                sampled_speech_token = torch.zeros(input_ids.shape[0], 0).to(input_ids.device)
                for i, speech_head in enumerate(self.speech_heads):
                    last_logits = speech_logits_1[i, :, -1] if i < 3 else speech_logits_2[i - 3, :, -1]
                    if repetition_penalty > 1.0:
                        if all([history_speech_tokens[i].shape[-1] != 0 for i in range(len(self.speech_heads))]):
                            score = last_logits.gather(1, history_speech_tokens[i].long())
                            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                            last_logits.scatter_(1, history_speech_tokens[i].long(), score)

                    if not do_sample:
                        sampled_tokens = self.get_greedy_token(
                            last_logits
                        )
                    else:
                        sampled_tokens = self.get_nucleus_one_token(
                            last_logits,
                            temperature=temperature,
                            top_p=top_p
                        )

                    history_speech_tokens[i] = torch.cat([history_speech_tokens[i], sampled_tokens], dim=1)
                    sampled_tokens = speech_head.mapping_sub_to_orig(sampled_tokens)
                    sampled_speech_token = torch.cat([sampled_speech_token, sampled_tokens], dim=1)

                # change inference mode and discard token
                if torch.any(sampled_speech_token[0, :] == eosp_idx):
                    inference_mode = "text"
                    new_token += 1

                    eosp_token = torch.tensor([eosp_idx], device=input_ids.device).unsqueeze(0)
                    input_ids = torch.cat([input_ids, eosp_token], dim=1).to(torch.long)
                    eosp_embeds = token_embed(eosp_token).flatten().unsqueeze(0).unsqueeze(0)
                    input_embeds = torch.cat([input_embeds, eosp_embeds], dim=1)
                elif torch.any(sampled_speech_token[0, 3:].long() == pad_idx[0]):
                    positions = torch.where(sampled_speech_token[0, :].long() == pad_idx[0])[0]  # [3,..]
                    first_pos = positions[0].item() // 3 * 3
                    new_token += (first_pos)

                    pad_token = torch.tensor([pad_idx[0]], device=input_ids.device).unsqueeze(0)
                    input_ids = torch.cat([input_ids, sampled_speech_token[:, :first_pos]], dim=1).to(torch.long)
                    next_embeds = token_embed(
                        torch.cat([sampled_speech_token[:, :first_pos].long(), pad_token.repeat(1, self.medusa - first_pos)],
                                  dim=1)
                    ).mean(dim=1).flatten().unsqueeze(0).unsqueeze(0)
                    input_embeds = torch.cat([input_embeds, next_embeds], dim=1)
                else:
                    new_token += sampled_speech_token.shape[-1]

                    input_ids = torch.cat([input_ids, sampled_speech_token], dim=1).to(torch.long)

                    next_embeds = token_embed(sampled_speech_token.long()).mean(dim=1).flatten().unsqueeze(0).unsqueeze(
                        0)
                    input_embeds = torch.cat([input_embeds, next_embeds], dim=1)
            past_key_values = outputs.past_key_values

            # eos_token_id = 151643
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id in input_ids[0, input_len:]:
                break

        return {
            "text": self.tokenizer.decode(
                input_ids[0, input_len:],
                skip_special_tokens=True,
            ),
            'new_token': new_token
        }

class MedusaModelForQwen2_3Head(MedusaModelForQwen2Base):
    def __init__(
            self,
            config: MedusaConfig,
    ):
        super().__init__(config)
    
    def forward(
            self,
            input_ids_a_1,
            input_ids_b_1,
            input_ids_c_1,
            label_a_1,
            label_b_1,
            label_c_1,
            speech_mask,
            spk_emb=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None,
            output_logit=False
    ):
        token_embed = self.lm_model.get_input_embeddings()

        text_embedding_a_1 = token_embed(input_ids_a_1)
        text_embedding_b_1 = token_embed(input_ids_b_1)
        text_embedding_c_1 = token_embed(input_ids_c_1)

        # replace the <|placeholder|> token text_embedding_a_1
        if self.spk_emb_proj:
            # (batch_size, seq_len)
            placeholder_mask =  (input_ids_a_1 == self.placeholder_index)
            spk_emb = self.spk_emb_proj(spk_emb)    # proj
            spk_emb_expanded = spk_emb.expand(-1, input_ids_a_1.shape[1], -1)   # expand
            
            text_embedding_a_1 = torch.where(
                placeholder_mask.unsqueeze(-1).expand(-1, -1, text_embedding_a_1.size(-1)).bool(),
                spk_emb_expanded,
                text_embedding_a_1
            )

        speech_embedding = torch.stack([text_embedding_a_1,
                                        text_embedding_b_1,
                                        text_embedding_c_1], dim=2).mean(dim=2)

        inputs_embeds = text_embedding_a_1 * (1 - speech_mask.unsqueeze(-1)) + speech_embedding * speech_mask.unsqueeze(
            -1)

        outputs = self.lm_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        # compute loss
        text_logits = self.lm_model.lm_head(outputs[0])
        label_a_mask = torch.where(speech_mask.bool(), -100, label_a_1)
        loss_text = self.loss_fct(text_logits[:, :-1, :].contiguous().view(-1, text_logits.shape[-1]),
                                  label_a_mask[:, 1:].contiguous().view(-1))

        hidden_states = outputs[0]

        if output_logit:
            ret_speech_logits_1 = []
            ret_speech_logits_2 = []

        head_losses = []
        head_top_one = []
        for layer, label in zip(self.speech_heads, [label_a_1, label_b_1, label_c_1]):
            speech_logits = layer(hidden_states)
            if output_logit:
                if layer.name in ["a1", "b1", "c1"]:
                    ret_speech_logits_1.append(speech_logits)
                elif layer.name in []:
                    ret_speech_logits_2.append(speech_logits)
                else:
                    raise ValueError(f"not support {layer.name}")
            head_loss, top_one_corr = self.compute_speech_loss(speech_logits, label, speech_mask, layer)
            head_losses.append(head_loss)
            head_top_one.append(top_one_corr)

        if output_logit:
            return text_logits, torch.stack(ret_speech_logits_1, dim=0), None

        loss_speech = sum(head_losses) / len(head_losses)

        loss = 0.2 * loss_text + 0.8 * loss_speech

        head_losses = [loss.item() for loss in head_losses]

        return loss, (loss_text, head_losses, loss_speech, head_top_one), outputs


class MedusaModelForQwen2_6Head(MedusaModelForQwen2Base):
    def __init__(
            self,
            config: MedusaConfig,
    ):
        super().__init__(config)
    
    def forward(
            self,
            input_ids_a_1,
            input_ids_a_2,
            input_ids_b_1,
            input_ids_b_2,
            input_ids_c_1,
            input_ids_c_2,
            label_a_1,
            label_a_2,
            label_b_1,
            label_b_2,
            label_c_1,
            label_c_2,
            speech_mask,
            spk_emb=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None,
            output_logit=False
    ):
        token_embed = self.lm_model.get_input_embeddings()

        text_embedding_a_1 = token_embed(input_ids_a_1)
        text_embedding_b_1 = token_embed(input_ids_b_1)
        text_embedding_c_1 = token_embed(input_ids_c_1)
        text_embedding_a_2 = token_embed(input_ids_a_2)
        text_embedding_b_2 = token_embed(input_ids_b_2)
        text_embedding_c_2 = token_embed(input_ids_c_2)

        if self.spk_emb_proj:
            # (batch_size, seq_len)
            placeholder_mask =  (input_ids_a_1 == self.placeholder_index)
            spk_emb = self.spk_emb_proj(spk_emb)    # proj
            spk_emb_expanded = spk_emb.expand(-1, input_ids_a_1.shape[1], -1)   # expand
            
            text_embedding_a_1 = torch.where(
                placeholder_mask.unsqueeze(-1).expand(-1, -1, text_embedding_a_1.size(-1)).bool(),
                spk_emb_expanded,
                text_embedding_a_1
            )

        speech_embedding = torch.stack([text_embedding_a_1,
                                        text_embedding_b_1,
                                        text_embedding_c_1,
                                        text_embedding_a_2,
                                        text_embedding_b_2,
                                        text_embedding_c_2], dim=2).mean(dim=2)

        inputs_embeds = text_embedding_a_1 * (1 - speech_mask.unsqueeze(-1)) + speech_embedding * speech_mask.unsqueeze(
            -1)
        

        outputs = self.lm_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        # compute loss
        text_logits = self.lm_model.lm_head(outputs[0])
        label_a_mask = torch.where(speech_mask.bool(), -100, label_a_1)
        loss_text = self.loss_fct(text_logits[:, :-1, :].contiguous().view(-1, text_logits.shape[-1]),
                                  label_a_mask[:, 1:].contiguous().view(-1))

        hidden_states = outputs[0]

        if output_logit:
            ret_speech_logits_1 = []
            ret_speech_logits_2 = []

        # head a1,b1,c1,a2,b2,c2
        head_losses = []
        head_top_one = []
        for layer, label in zip(self.speech_heads, [label_a_1, label_b_1, label_c_1,
                                                    label_a_2, label_b_2, label_c_2]):
            speech_logits = layer(hidden_states)
            if output_logit:
                if layer.name in ["a1", "b1", "c1"]:
                    ret_speech_logits_1.append(speech_logits)
                elif layer.name in ["a2", "b2", "c2"]:
                    ret_speech_logits_2.append(speech_logits)
                else:
                    raise ValueError(f"not support {layer.name}")
            head_loss, top_one_corr = self.compute_speech_loss(speech_logits, label, speech_mask, layer)
            head_losses.append(head_loss)
            head_top_one.append(top_one_corr)

        if output_logit:
            return text_logits, torch.stack(ret_speech_logits_1, dim=0), torch.stack(ret_speech_logits_2, dim=0)

        loss_speech = sum(head_losses) / len(head_losses)

        loss = 0.2 * loss_text + 0.8 * loss_speech

        head_losses = [loss.item() for loss in head_losses]

        return loss, (loss_text, head_losses, loss_speech, head_top_one), outputs


class MedusaModelForQwen2_12Head(MedusaModelForQwen2Base):
    def __init__(
            self,
            config: MedusaConfig,
    ):
        super().__init__(config)
    
    def forward(
            self,
            input_ids_a_1,
            input_ids_a_2,
            input_ids_a_3,
            input_ids_a_4,
            input_ids_b_1,
            input_ids_b_2,
            input_ids_b_3,
            input_ids_b_4,
            input_ids_c_1,
            input_ids_c_2,
            input_ids_c_3,
            input_ids_c_4,
            label_a_1,
            label_a_2,
            label_a_3,
            label_a_4,
            label_b_1,
            label_b_2,
            label_b_3,
            label_b_4,
            label_c_1,
            label_c_2,
            label_c_3,
            label_c_4,
            speech_mask,
            spk_emb=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None,
            output_logit=False
    ):
        token_embed = self.lm_model.get_input_embeddings()

        text_embedding_a_1 = token_embed(input_ids_a_1)
        text_embedding_b_1 = token_embed(input_ids_b_1)
        text_embedding_c_1 = token_embed(input_ids_c_1)
        text_embedding_a_2 = token_embed(input_ids_a_2)
        text_embedding_b_2 = token_embed(input_ids_b_2)
        text_embedding_c_2 = token_embed(input_ids_c_2)
        text_embedding_a_3 = token_embed(input_ids_a_3)
        text_embedding_b_3 = token_embed(input_ids_b_3)
        text_embedding_c_3 = token_embed(input_ids_c_3)
        text_embedding_a_4 = token_embed(input_ids_a_4)
        text_embedding_b_4 = token_embed(input_ids_b_4)
        text_embedding_c_4 = token_embed(input_ids_c_4)

        # replace the <|placeholder|> token text_embedding_a_1
        if self.spk_emb_proj:
            # (batch_size, seq_len)
            placeholder_mask =  (input_ids_a_1 == self.placeholder_index)
            spk_emb = self.spk_emb_proj(spk_emb)    # proj
            spk_emb_expanded = spk_emb.expand(-1, input_ids_a_1.shape[1], -1)   # expand
            
            text_embedding_a_1 = torch.where(
                placeholder_mask.unsqueeze(-1).expand(-1, -1, text_embedding_a_1.size(-1)).bool(),
                spk_emb_expanded,
                text_embedding_a_1
            )

        speech_embedding = torch.stack([text_embedding_a_1,
                                        text_embedding_b_1,
                                        text_embedding_c_1,
                                        text_embedding_a_2,
                                        text_embedding_b_2,
                                        text_embedding_c_2,
                                        text_embedding_a_3,
                                        text_embedding_b_3,
                                        text_embedding_c_3,
                                        text_embedding_a_4,
                                        text_embedding_b_4,
                                        text_embedding_c_4], dim=2).mean(dim=2)

        inputs_embeds = text_embedding_a_1 * (1 - speech_mask.unsqueeze(-1)) + speech_embedding * speech_mask.unsqueeze(
            -1)

        outputs = self.lm_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        # compute loss
        text_logits = self.lm_model.lm_head(outputs[0])
        label_a_mask = torch.where(speech_mask.bool(), -100, label_a_1)
        loss_text = self.loss_fct(text_logits[:, :-1, :].contiguous().view(-1, text_logits.shape[-1]),
                                  label_a_mask[:, 1:].contiguous().view(-1))

        hidden_states = outputs[0]

        if output_logit:
            ret_speech_logits_1 = []
            ret_speech_logits_2 = []

        # head a1,b1,c1,a2,b2,c2
        head_losses = []
        head_top_one = []
        for layer, label in zip(self.speech_heads, [label_a_1, label_b_1, label_c_1,
                                                    label_a_2, label_b_2, label_c_2,
                                                    label_a_3, label_b_3, label_c_3,
                                                    label_a_4, label_b_4, label_c_4]):
            speech_logits = layer(hidden_states)
            if output_logit:
                if layer.name in ["a1", "b1", "c1"]:
                    ret_speech_logits_1.append(speech_logits)
                elif layer.name in ["a2", "b2", "c2", "a3", "b3", "c3", "a4", "b4", "c4"]:
                    ret_speech_logits_2.append(speech_logits)
                else:
                    raise ValueError(f"not support {layer.name}")
            head_loss, top_one_corr = self.compute_speech_loss(speech_logits, label, speech_mask, layer)
            head_losses.append(head_loss)
            head_top_one.append(top_one_corr)

        if output_logit:
            return text_logits, torch.stack(ret_speech_logits_1, dim=0), torch.stack(ret_speech_logits_2, dim=0)

        loss_speech = sum(head_losses) / len(head_losses)

        loss = 0.2 * loss_text + 0.8 * loss_speech

        head_losses = [loss.item() for loss in head_losses]

        return loss, (loss_text, head_losses, loss_speech, head_top_one), outputs