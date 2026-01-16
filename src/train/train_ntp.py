import logging
import os
import sys
import math
import argparse

from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import HfArgumentParser, TrainingArguments, AutoConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer

from ..utils.prompter import Prompter, normalize_token, normalize_text
from ..utils.collator.ntp import DataCollator

from ..model.tts.modeling_ntp import ModelForQwen2
from ..model.tts.configuration import ModelConfig



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SpeechTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)

        if return_outputs:
            return outputs.loss, outputs
        
        return outputs.loss

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
    )
    spk_emb_dim: Optional[int] = field(
        default=256
    )
    
@dataclass
class DataArguments:
    data_path: str = field(
        default="", 
        metadata={"help": "Path to the training data."})
    text_data_path: str = field(
        default="", 
        metadata={"help": "Path to the text training data."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    spk_aware: bool = field(
        default=False, metadata={"help": "spk_aware"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    model_max_length: int = field(
        # default=4096,
        default=32768,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    val_set_size: int = field(
        default=0,
        metadata={"help": "val_set_size"},
    )
    preprocessing_num_workers: int = field(
        default=100,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )
    initial_global_step: int = field(
        default=0,
        metadata={"help": "initial_global_step"}
    )
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(cfg.cfg_file)
    os.makedirs(training_args.cache_dir, exist_ok=True)
    # parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    prompter = Prompter()

    pretrained_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config = ModelConfig(
        base_model_name_or_path=model_args.model_name_or_path,
        spk_emb_dim=model_args.spk_emb_dim,
        model_max_length=training_args.model_max_length,
        **pretrained_config.to_dict()
    )

    model = ModelForQwen2(config)
    
    tokenizer = model.tokenizer
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def tokenize(prompt, add_eos_token=True, add_bos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        prompts = prompt.split('[Assistant]:')
        
        for i in range(len(prompts)-1):
            prompts[i] += '[Assistant]:'
        results = tokenizer(prompts[0], truncation=True, max_length=tokenizer.model_max_length, padding=False, return_tensors=None, add_special_tokens=False)
        results["labels"] = [-100] * len(results["input_ids"])        
        
        # print(results["input_ids"])

        for prompt in prompts[1:-1]:
            gpt, human = prompt.split('[Human]:')
            human = '[Human]:' + human
            
            result = tokenizer(gpt, truncation=True, max_length=tokenizer.model_max_length, padding=False, return_tensors=None, add_special_tokens=False)
            results["input_ids"].extend(result["input_ids"])
            results["attention_mask"].extend(result["attention_mask"])
            results["labels"].extend(result["input_ids"].copy())

            result = tokenizer(human, truncation=True, max_length=tokenizer.model_max_length, padding=False, return_tensors=None, add_special_tokens=False)
            results["input_ids"].extend(result["input_ids"])
            results["attention_mask"].extend(result["attention_mask"])
            results["labels"].extend([-100] * len(result["input_ids"]))


        result = tokenizer(prompts[-1], truncation=True, max_length=tokenizer.model_max_length, padding=False, return_tensors=None, add_special_tokens=False)
        results["input_ids"].extend(result["input_ids"])
        results["attention_mask"].extend(result["attention_mask"])
        results["labels"].extend(result["input_ids"].copy())

        if (
            results["input_ids"][-1] != tokenizer.eos_token_id
            and add_eos_token
        ):
            results["input_ids"].append(tokenizer.eos_token_id)
            results["attention_mask"].append(1)
            results["labels"].append(tokenizer.eos_token_id)

        if (
            results["input_ids"][0] != tokenizer.bos_token_id
            and add_bos_token
        ):
            results["input_ids"].insert(0, tokenizer.bos_token_id)
            results["attention_mask"].insert(0, 1)
            results["labels"].insert(0, -100)
        
        if len(results["input_ids"]) > tokenizer.model_max_length:
            results["input_ids"] = results["input_ids"][:tokenizer.model_max_length-1] + [tokenizer.eos_token_id]
            results["attention_mask"] = results["attention_mask"][:tokenizer.model_max_length-1] + [1]
            results["labels"] = results["labels"][:tokenizer.model_max_length-1] + [tokenizer.eos_token_id]
            
        return results


    def generate_and_tokenize_prompt_tts(data_point):
        full_prompt_tts = prompter.generate_prompt(
            task="tts",
            text=normalize_text(data_point["text"]),
            audio=normalize_token(data_point["tokens"]),
            spk_aware=data_args.spk_aware
        )

        tokenized_full_prompt = tokenize(full_prompt_tts)
        spk_emb_path = data_point.get("spk_emb_path", None)

        return {
            "spk_emb_path": spk_emb_path,
            **tokenized_full_prompt,
        }


    def generate_and_tokenize_prompt_text(data_point):
        data_list = data_point["conversations"]
        instructions = data_list[::2]
        responses = data_list[1::2]
        
        full_prompt_text = ""
        for _, (inst, resp) in enumerate(zip(instructions, responses)):
            full_prompt_text += prompter.generate_prompt(
                                        task="text",
                                        instruction=inst["value"],
                                        response=resp["value"],
                                    )

        tokenized_full_prompt = tokenize(full_prompt_text)
        return tokenized_full_prompt

    # load data
    data_path = data_args.data_path
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)
    
    # Tokenize
    datasets_to_tokenize = [
        data["train"].map(
            generate_and_tokenize_prompt_tts,
            batched=False,
            num_proc=training_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            cache_file_name=os.path.join(training_args.cache_dir, 'tokenized', 'tts', 'processed_train.arrow')
        ).select_columns(["input_ids", "labels", "attention_mask", "spk_emb_path"])
    ]
    
    # load text data
    text_data_path = data_args.text_data_path
    if text_data_path != "":
        text_data = load_dataset("json", data_files=text_data_path)
    else:
        text_data = None
        print("No text data path provided. Skipping text data loading.")
        
    if text_data is not None:   
        datasets_to_tokenize.append(
            text_data["train"].map(
                generate_and_tokenize_prompt_text,
                batched=False,
                num_proc=training_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=os.path.join(training_args.cache_dir, 'tokenized', 'text', 'processed_train.arrow')
            ).select_columns(["input_ids", "labels", "attention_mask"])
        )
    
    tokenized_dataset = concatenate_datasets(datasets_to_tokenize).shuffle()
    
    if training_args.val_set_size > 0:
        train_val = tokenized_dataset.train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        train_data = tokenized_dataset
        val_data = None
        
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        return examples
    
    group_cache_file_name = os.path.join(training_args.cache_dir, 'group', 'train', 'processed_train.arrow')
    
    
    with training_args.main_process_first(desc="grouping texts together"):
        train_data = train_data.map(
            group_texts,
            batched=True,
            num_proc=5,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Without group",
            cache_file_name=group_cache_file_name,
        )
        

    data_collator = DataCollator(
        tokenizer
    )

    # empty cache
    torch.cuda.empty_cache()

    trainer = SpeechTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_data if training_args.do_train else None, 
        data_collator=data_collator
        )
    
    # empty cache
    torch.cuda.empty_cache()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(val_data)
        metrics["eval_samples"] = min(max_eval_samples, len(val_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    cfg_parser = argparse.ArgumentParser()
    cfg_parser.add_argument("--cfg_file", type=str, required=True)
    cfg = cfg_parser.parse_args()
    train()