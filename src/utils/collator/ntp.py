from transformers import PreTrainedTokenizerBase
from typing import Optional, Union
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
import torch
import os

@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    spk_aware: bool = False

    def __call__(self, features, return_tensors=None):

        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        spk_emb_list = []

        max_input_len_list = []
        for feature in features:
            # print(feature["spk_emb_path"])
            input_ids_list.append(feature["input_ids"])
            labels_list.append(feature["labels"])
            attention_mask_list.append(feature["attention_mask"])
            if self.spk_aware:
                spk_emb_list.append(torch.load(os.path.join(self.dataset_root, feature['spk_emb_path'][3])).to(torch.bfloat16))


            max_input_len_list.append(len(feature["input_ids"]))

        max_input_len = max(max_input_len_list)

        if self.tokenizer.padding_side == "left":
            final_input_ids = torch.stack(
                [
                    torch.cat([
                        torch.full((max_input_len - max_input_len_list[idx], ), self.tokenizer.pad_token_id),
                        torch.tensor(input_ids)
                    ], dim=0)   # cat on seq len
                for idx, input_ids in enumerate(input_ids_list)], dim=0    # stack on batch
            )
            final_labels = torch.stack(
                [
                    torch.cat([
                        torch.full((max_input_len - max_input_len_list[idx], ), self.label_pad_token_id),
                        torch.tensor(label)
                    ], dim=0)
                for idx, label in enumerate(labels_list)], dim=0
            )

            attention_mask = torch.stack(
                [
                    torch.cat([
                        torch.full((max_input_len - max_input_len_list[idx], ), 0),
                        torch.tensor(mask)
                    ], dim=0)
                for idx, mask in enumerate(attention_mask_list)], dim=0
            )

        elif self.tokenizer.padding_side == "right":
            final_input_ids = torch.stack(
                [
                    torch.cat([
                        torch.tensor(input_ids),
                        torch.full((max_input_len - max_input_len_list[idx], ), self.tokenizer.pad_token_id)
                    ], dim=0)   # cat on seq len
                for idx, input_ids in enumerate(input_ids_list)], dim=0    # stack on batch
            )
            final_labels = torch.stack(
                [
                    torch.cat([
                        torch.tensor(label),
                        torch.full((max_input_len - max_input_len_list[idx], ), self.label_pad_token_id)
                    ], dim=0)
                for idx, label in enumerate(labels_list)], dim=0
            )

            attention_mask = torch.stack(
                [
                    torch.cat([
                        torch.tensor(mask),
                        torch.full((max_input_len - max_input_len_list[idx], ), 0)
                    ], dim=0)
                for idx, mask in enumerate(attention_mask_list)], dim=0
            )

        else:
            raise ValueError(f"padding side == {self.tokenizer.padding_side} is not implemented!")
        

        # speaker embedding for batch
        if self.spk_aware:
            spk_emb = torch.stack(spk_emb_list, dim=0)
        else:
            spk_emb = None # WARNING: need to check none when using


        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "attention_mask": attention_mask,
            "spk_emb": spk_emb
        }