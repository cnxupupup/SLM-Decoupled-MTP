from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import HfArgumentParser, TrainingArguments, AutoConfig
from transformers.trainer_utils import get_last_checkpoint
import os
import logging
import sys
import math

from ..utils.prompter import Prompter, normalize_token, normalize_text

# for multi token prediction

from transformers import Trainer

from ..model.tts.modeling_mtp import (MedusaModelForQwen2_3Head, 
                                      MedusaModelForQwen2_6Head, 
                                      MedusaModelForQwen2_12Head
                                      )
from ..utils.collator.mtp import DataCollatorForSpeech
from ..model.tts.configuration_mtp import MedusaConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SpeechTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, return_log=False, do_log=False):
        loss, (loss_text, medusa_losses, loss_speech, medusa_top_one), outputs = model(**inputs)
        log = {"loss_compute_loss": loss.item(), "loss_text": loss_text.item()}
        if medusa_losses is not None:
            log['loss_speech'] = loss_speech.item()
            for i, medusa_loss, medusa_top_one_i in zip(range(len(medusa_losses)), medusa_losses, medusa_top_one):
                log[f"medusa_loss_{i}"] = medusa_loss
                log[f"medusa_top_one_{i}"] = medusa_top_one_i
        if do_log:
            self.log(log)

        if return_log:
            return loss, log
        if return_outputs:
            return loss, outputs

        return loss


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
    head_implement: Optional[str] = field(
        default="linear"
    )
    head_transformers_layer: Optional[int] = field(
        default=2
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
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"},
    )
    # for multi token prediction
    num_medusa_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads."},
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
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    config = MedusaConfig(medusa_num_heads=training_args.medusa_num_heads,
                          medusa_num_layers=training_args.medusa_num_layers,
                          base_model_name_or_path=model_args.model_name_or_path,
                          model_max_length=training_args.model_max_length,
                          head_implement=model_args.head_implement,
                          head_transformers_layer=model_args.head_transformers_layer,
                          **pretrained_config.to_dict())
    if training_args.num_medusa_heads == 3:
        medusa_lm_head = MedusaModelForQwen2_3Head(config)
    elif training_args.num_medusa_heads == 6:
        medusa_lm_head = MedusaModelForQwen2_6Head(config)
    elif training_args.num_medusa_heads == 12:
        medusa_lm_head = MedusaModelForQwen2_12Head(config)
    else:
        raise NotImplementedError(f"{training_args.num_medusa_heads} heads is not supported for now!")


    tokenizer = medusa_lm_head.tokenizer

    def tokenize(prompt, add_eos_token=True, add_bos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        prompts = prompt.split('[Assistant]:')

        for i in range(len(prompts) - 1):
            prompts[i] += '[Assistant]:'
        results = tokenizer(prompts[0], truncation=True, max_length=tokenizer.model_max_length, padding=False,
                            return_tensors=None, add_special_tokens=False)
        results["labels"] = [-100] * len(results["input_ids"])

        for prompt in prompts[1:-1]:
            gpt, human = prompt.split('[Human]:')
            human = '[Human]:' + human

            result = tokenizer(gpt, truncation=True, max_length=tokenizer.model_max_length, padding=False,
                               return_tensors=None, add_special_tokens=False)
            results["input_ids"].extend(result["input_ids"])
            results["attention_mask"].extend(result["attention_mask"])
            results["labels"].extend(result["input_ids"].copy())

            result = tokenizer(human, truncation=True, max_length=tokenizer.model_max_length, padding=False,
                               return_tensors=None, add_special_tokens=False)
            results["input_ids"].extend(result["input_ids"])
            results["attention_mask"].extend(result["attention_mask"])
            results["labels"].extend([-100] * len(result["input_ids"]))

        result = tokenizer(prompts[-1], truncation=True, max_length=tokenizer.model_max_length, padding=False,
                           return_tensors=None, add_special_tokens=False)
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
            results["input_ids"] = results["input_ids"][:tokenizer.model_max_length - 1] + [tokenizer.eos_token_id]
            results["attention_mask"] = results["attention_mask"][:tokenizer.model_max_length - 1] + [1]
            results["labels"] = results["labels"][:tokenizer.model_max_length - 1] + [tokenizer.eos_token_id]

        return results

    sosp_idx, eosp_idx = medusa_lm_head.get_sosp_eosp_idx()

    def generate_and_tokenize_prompt_tts(data_point):
        full_prompts = {}
        for letter in ['a', 'b', 'c']:
            for i in range(1, 2):
                full_prompts[f"{letter}_{i}"] = prompter.generate_prompt(
                    task="tts",
                    text=normalize_text(data_point["text"]),
                    audio=normalize_token(data_point["tokens"][f"head_{letter}_{i}"]),
                    spk_aware=data_args.spk_aware,
                )

        tokenized = {}
        for key in full_prompts:
            tokenized[key] = tokenize(full_prompts[key])

        def generate_mask(input_ids):
            mask = []
            in_between = False
            for item in input_ids:
                if item == sosp_idx:
                    mask.append(0)
                    in_between = True
                elif item == eosp_idx:
                    mask.append(1)
                    in_between = False
                elif in_between:
                    mask.append(1)
                else:
                    mask.append(0)
            return mask

        # Prepare the output dictionary with all 48 heads
        tokenized_full_prompt = {
            "attention_mask": tokenized["a_1"]["attention_mask"],
            "speech_mask": generate_mask(tokenized["a_1"]["input_ids"])
        }
        # Add all input_ids and labels
        for letter in ['a', 'b', 'c']:
            for i in range(1, training_args.num_medusa_heads // 3 + 1):
                tokenized_full_prompt[f"input_ids_{letter}_{i}"] = tokenized[f"{letter}_{i}"]["input_ids"]
                tokenized_full_prompt[f"label_{letter}_{i}"] = tokenized[f"{letter}_{i}"]["labels"]
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
        ).select_columns(
            [f"input_ids_a_{i}" for i in range(1, 2)] + [f"input_ids_b_{i}" for i in range(1, 2)] + [f"input_ids_c_{i}"
                                                                                                     for i in
                                                                                                     range(1, 2)] + \
            [f"label_a_{i}" for i in range(1, 2)] + [f"label_b_{i}" for i in range(1, 2)] + [f"label_c_{i}" for i in
                                                                                             range(1, 2)] + \
            ["speech_mask", "attention_mask"])
    ]

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
            num_proc=20,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Without group",
            cache_file_name=group_cache_file_name,
        )

    data_collator = DataCollatorForSpeech(tokenizer, 
                                          num_heads=training_args.num_medusa_heads,
                                          spk_aware=data_args.spk_aware)

    # empty cache
    torch.cuda.empty_cache()

    trainer = SpeechTrainer(
        model=medusa_lm_head,
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
    train()