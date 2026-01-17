from transformers import Qwen2PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch
from safetensors.torch import load_file
import os
from torch.nn import functional as F
import glob

from .configuration_ntp import ModelConfig


class PretrainedModelForQwen2(Qwen2PreTrainedModel):
    config_class = ModelConfig

class ModelForQwen2(PretrainedModelForQwen2):
    def __init__(self, config: ModelConfig, special_tokens=['<sosp>','<eosp>','[Human]','[Assistant]', '<|placeholder|>', '[ta]', '[ua]']):
        super().__init__(config)
        # self.loss_function = CrossEntropyLoss(ignore_index=-100)
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
        self._add_tokens(special_tokens)
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
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        special_tokens=['<sosp>','<eosp>','[Human]','[Assistant]', '<|placeholder|>', '[ta]', '[ua]'],
        **kwargs
    ):
        config = ModelConfig.from_pretrained(model_name_or_path)
        model = cls(config, special_tokens, **kwargs)
        possible_path = os.path.join(model_name_or_path, "model.safetensors")
        if os.path.exists(possible_path):
            state_dict = load_file(possible_path)
            if "lm_model.model.embed_tokens.weight" in state_dict:
                state_dict["lm_model.lm_head.weight"] = state_dict["lm_model.model.embed_tokens.weight"]
            model.load_state_dict(state_dict)

        else:
            state_dict = {}
            paths = glob.glob(os.path.join(model_name_or_path, "*.safetensors"))
            for path in paths:
                state_part = load_file(path)
                state_dict.update(state_part)
            if "lm_model.model.embed_tokens.weight" in state_dict:
                state_dict["lm_model.lm_head.weight"] = state_dict["lm_model.model.embed_tokens.weight"]

            model.load_state_dict(state_dict)

        return model
    

    def _add_tokens(self, special_tokens=['<sosp>','<eosp>','[Human]','[Assistant]', '<|placeholder|>', '[ta]', '[ua]']):
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
        # add special token
        if "<sosp>" not in self.tokenizer.get_vocab():
            units_size = 1024
            new_tokens = [f"<a{x}>" for x in range(units_size)] + \
                        [f"<b{x}>" for x in range(units_size)] + \
                        [f"<c{x}>" for x in range(units_size)] + special_tokens
            self.tokenizer.add_tokens(new_tokens)

        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([token])

        # resize embedding
        embedding_size = self.lm_model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.lm_model.resize_token_embeddings(len(self.tokenizer))

    def forward(
            self,
            input_ids,
            labels,
            spk_emb=None,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            position_ids=None,
    ):  
        if self.spk_emb_proj:
            # (batch_size, seq_len)
            placeholder_mask =  (input_ids == self.placeholder_index)
            spk_emb = self.spk_emb_proj(spk_emb)    # proj
            spk_emb_expanded = spk_emb.expand(-1, input_ids.shape[1], -1)   # expand
            input_embs = self.lm_model.get_input_embeddings()(input_ids)
            
            input_embs = torch.where(
                placeholder_mask.unsqueeze(-1).expand(-1, -1, input_embs.size(-1)).bool(),
                spk_emb_expanded,
                input_embs
            )
        else:
            input_embs = self.lm_model.get_input_embeddings()(input_ids)
        return self.lm_model(
            inputs_embeds=input_embs,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            position_ids=position_ids
        )
    
    def get_nucleus_one_token(self, logit, temperature, top_p):
        if top_p >= 1:
            return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
        logit = logit / temperature
        probs = torch.softmax(logit, dim=-1)
        sorted_logits, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_logits, dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logit[indices_to_remove] = float('-inf')
        sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
        return sampled_tokens

    def get_greedy_token(self, logits):
        _, sampled_tokens = torch.max(F.softmax(logits, dim=-1), dim=-1)
        return sampled_tokens.unsqueeze(1)
    
    @torch.inference_mode()
    def generate(self, 
                input_ids,
                spk_emb,
                do_sample=False,
                use_cache=True,
                temperature_text=1.0,
                temperature_speech=1.0,
                repetition_penalty=1.0,
                top_p=0.8,
                max_new_tokens=512):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now !!"
        past_key_values = None
        token_embed = self.lm_model.get_input_embeddings()
        history_tokens = None
        sosp_idx = self.tokenizer.convert_tokens_to_ids("<sosp>")
        
        input_len = input_ids.shape[1]
        assert input_len > 0, "input length must > 0"
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
        inference_mode = 'text'
        for _ in range(max_new_tokens):
            outputs = self.lm_model.model(
                inputs_embeds=input_embs if past_key_values is None else input_embs[:, -1:],
                use_cache=use_cache,
                past_key_values=past_key_values
            )
            hidden_state = outputs[0]
            logits = self.lm_model.lm_head(hidden_state)
            last_logits = logits[:, -1]
            if repetition_penalty > 1.0:
                if history_tokens is not None:
                    score = last_logits.gather(1, history_tokens)
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    last_logits.scatter_(1, history_tokens, score)
            if not do_sample:
                sampled_tokens = self.get_greedy_token(
                    last_logits
                )
            else:
                if inference_mode == 'text':
                    sampled_tokens = self.get_nucleus_one_token(
                        last_logits,
                        temperature=temperature_text, 
                        top_p=top_p
                    )
                else:
                    sampled_tokens = self.get_nucleus_one_token(
                        last_logits,
                        temperature=temperature_speech, 
                        top_p=top_p
                    )
            # print(self.tokenizer.batch_decode(sampled_tokens.cpu()))
            if history_tokens is None:
                history_tokens = sampled_tokens.long()
            else:
                history_tokens = torch.cat([history_tokens, sampled_tokens.long()], dim=1)

            input_ids = torch.cat([input_ids, sampled_tokens], dim=1).to(torch.long)
            input_embs = torch.cat([input_embs, token_embed(sampled_tokens)], dim=1)
            past_key_values = outputs.past_key_values
            if sampled_tokens.cpu().flatten().item() == sosp_idx:
                inference_mode = 'speech'
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
        return self.tokenizer.batch_decode(input_ids[:, input_len:].cpu(), skip_special_tokens=True)[0]






