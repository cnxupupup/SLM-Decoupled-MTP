from transformers import PreTrainedTokenizerBase
from typing import Optional, Union
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
import torch
import os


@dataclass
class DataCollatorForSpeech:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    num_heads: int = 3
    spk_aware: bool = False

    def __call__(self, features, return_tensors=None):

        input_ids_lists = {f"{letter}{i}": [] for letter in ['a', 'b', 'c'] for i in range(1, self.num_heads // 3 + 1)}
        label_lists = {f"{letter}{i}": [] for letter in ['a', 'b', 'c'] for i in range(1, self.num_heads // 3 + 1)}

        speech_mask_list = []
        spk_emb_list = []
        attention_mask_list = []
        max_input_len_list = []

        for feature in features:
            # Collect input_ids for all 48 heads
            for letter in ['a', 'b', 'c']:
                for i in range(1, self.num_heads // 3 + 1):
                    input_ids_lists[f"{letter}{i}"].append(feature[f"input_ids_{letter}_{i}"])

            # Collect labels for all 48 heads
            for letter in ['a', 'b', 'c']:
                for i in range(1, self.num_heads // 3 + 1):
                    label_lists[f"{letter}{i}"].append(feature[f"label_{letter}_{i}"])

            speech_mask_list.append(feature["speech_mask"])
            attention_mask_list.append(feature["attention_mask"])
            max_input_len_list.append(len(feature["input_ids_a_1"]))
            if self.spk_aware:
                spk_emb_list.append(torch.load(os.path.join(self.dataset_root, feature['spk_emb_path'][3])).to(torch.bfloat16))

        max_input_len = max(max_input_len_list)

        if self.tokenizer.padding_side == "left":
            raise NotImplementedError("left padding is not supported")

        elif self.tokenizer.padding_side == "right":

            final_input_ids = {}
            for key in input_ids_lists:
                final_input_ids[key] = torch.stack([
                    torch.cat([
                        torch.tensor(input_ids),
                        torch.full((max_input_len - max_input_len_list[idx],), self.tokenizer.pad_token_id)
                    ], dim=0)
                    for idx, input_ids in enumerate(input_ids_lists[key])
                ], dim=0)

            final_labels = {}
            for key in label_lists:
                final_labels[key] = torch.stack([
                    torch.cat([
                        torch.tensor(label),
                        torch.full((max_input_len - max_input_len_list[idx],), self.label_pad_token_id)
                    ], dim=0)
                    for idx, label in enumerate(label_lists[key])
                ], dim=0)

            attention_mask = torch.stack(
                [
                    torch.cat([
                        torch.tensor(mask),
                        torch.full((max_input_len - max_input_len_list[idx],), 0)
                    ], dim=0)
                    for idx, mask in enumerate(attention_mask_list)], dim=0
            )

            final_speech_mask = torch.stack(
                [
                    torch.cat([
                        torch.tensor(speech_mask),
                        torch.full((max_input_len - max_input_len_list[idx],), 0)
                    ], dim=0)
                    for idx, speech_mask in enumerate(speech_mask_list)], dim=0
            )

            # Verify all tensors have the same shape
            ref_shape = final_input_ids["a1"].shape
            for key in final_input_ids:
                assert final_input_ids[key].shape == ref_shape, f"input_ids_{key} shape mismatch"
            for key in final_labels:
                assert final_labels[key].shape == ref_shape, f"label_{key} shape mismatch"
            assert final_speech_mask.shape == ref_shape, "speech_mask shape mismatch"
            assert attention_mask.shape == ref_shape, "attention_mask shape mismatch"

        else:
            raise ValueError(f"padding side == {self.tokenizer.padding_side} is not implemented!")
        if self.spk_aware:
            spk_emb = torch.stack(spk_emb_list, dim=0)
        else:
            spk_emb = None # WARNING: need to check none when using
        # Prepare the return dictionary
        result = {
            "speech_mask": final_speech_mask,
            "attention_mask": attention_mask,
            "spk_emb": spk_emb
        }

        # Add all input_ids and labels to the result dictionary
        for letter in ['a', 'b', 'c']:
            for i in range(1, self.num_heads // 3 + 1):
                result[f"input_ids_{letter}_{i}"] = final_input_ids[f"{letter}{i}"]
                result[f"label_{letter}_{i}"] = final_labels[f"{letter}{i}"]

        return result