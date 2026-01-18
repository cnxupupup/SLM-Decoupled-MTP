# (AAAI '26) What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study
[![arXiv](https://img.shields.io/badge/arXiv-2506.12537-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.12537)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://cnxupupup.github.io/SLM-Decoupled-MTP-Demo)
[![RoleTriviaQA](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-RoleTriviaQA-ffbd45)](https://huggingface.co/datasets/cnxup/RoleTriviaQA)


Official implementation of paper [What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study](https://arxiv.org/abs/2506.12537).

## News
**[2026.01.18]** The inference code has been released

**[2026.01.17]** The MTP training code & demo data has been released.

**[2026.01.16]** The NTP training code has been released.

## Overview
Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective crossmodal alignment and high-quality speech generation. In this work, we systematically investigate the role of speech tokenizer designs in LLM-centric SLMs, augmented by speech heads and speaker modeling. We compare coupled, semidecoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12× faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.

## TODOs
- [x] Architecture code of the NTP, MTP and Role-Playing Knowledge Speech QA task w/ and w/o speaker aware.
- [x] RoleTriviaQA dataset
- [x] Training code
- [x] Inference code

## Environment Setup
You can install the environment by:
```bash
conda create -n slm python=3.10
conda activate slm
pip install -r requirements.txt
```

## Pre-requisites
### Models
- Download [FACodec](https://huggingface.co/amphion/naturalspeech3_facodec) for decoupled tokenization. We utilized `amphion/naturalspeech3_facodec/blob/main/ns3_facodec_encoder_v2.bin` as the encoder and `amphion/naturalspeech3_facodec/blob/main/ns3_facodec_decoder_v2.bin` as the decoder.

- Download [Qwen-2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) for NTP and MTP Text-to-Speech fine-tuning.

- Download [Qwen-2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) for Role-Playing Knowledge QA pre-training and fine-tuning.

### Datasets
- Please download [LibriTTS](https://www.openslr.org/60) for TTS fine-tuning

- Please download [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) for Role-Playing Knowledge QA pre-training. We filtered data samples with DNSMOS score >= 3.5.

- Please download [RoleTriviaQA](https://huggingface.co/datasets/cnxup/RoleTriviaQA) for Role-Playing Knowledge QA fine-tuning.

## Training

### Training NTP TTS models
We provide NTP demo data tokenized by FACodec w.o. speaker-aware in `demo_data/slm-ntp-data/example.jsonl`

To train the NTP model, you need to

1. modify `config/mtp_demo.yaml` similarly with the NTP config:
```yaml
# ModelArguments
model_name_or_path: "YOUR_PATH_TO_QWEN2.5_0.5B"
attn_implementation: "flash_attention_2"
spk_emb_dim: 256 # unused if "spk_aware" is false

# DataArguments
data_path: "demo_data/slm-ntp-data/example.jsonl"
text_data_path: "" # only used if text data is needed at pre-training stage
spk_aware: false

# TrainingArguments
bf16: true
output_dir: "YOUR_DIR_TO_SAVE_CKPTS"
cache_dir: "YOUR_DIR_TO_SAVE_TOKENIZED_DATA"
do_train: true
do_eval: true
val_set_size: 2
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
max_steps: 1000
eval_strategy: "steps"
eval_steps: 100
save_strategy: "steps"
save_steps: 100
learning_rate: 5e-4
weight_decay: 0.0
warmup_ratio: 0.03
lr_scheduler_type: "cosine"
log_level: debug
logging_strategy: "steps"
logging_steps: 1
overwrite_output_dir: false
remove_unused_columns: false
ddp_timeout: 5400
```
2. run `sh scripts/train_ntp.sh`

### Training MTP TTS models
We provide MTP demo data tokenized by FACodec for 3, 6 and 12 heads w.o. speaker-aware in `demo_data/slm-mtp-data/*.jsonl`

To train the MTP model, you need to 

1. modify `config/mtp_demo.yaml` similarly with the NTP config. 
**Note that you have to modify `num_medusa_heads` for different number of MTP heads**.
2. run `sh scripts/train_mtp.sh`


## Inference
We provide inference code for NTP and MTP with 3 heads w.o. speaker-aware
### NTP
1. Modify the parameters to your own paths

```sh
SAVEROOT="YOUR_SAVE_ROOT"
ckpt_step=54
mkdir -p $SAVEROOT
mkdir -p $SAVEROOT/logs
python infer.py \
--model-path "YOUR_MODEL_PATH" \
--save-dir "$SAVEROOT/${ckpt_step}k" \   # the directory to save the tts result
--logger-path "$SAVEROOT/logs/${ckpt_step}k.log" \ 
--text-path "YOUR_DIR_OF_TEXT" \
--wav-ref-path "YOUR_WAV_REF_PATH" # this is needed for FACodec inference \ 
--task "tts"
```

2. Then run
```sh
sh scripts/infer_ntp.sh
```

### MTP

1. Modify the model path in `src/infer/tts/infer_mtp.py`

2. Run 
```sh
python src/infer/tts/infer_mtp.py
```



## Citation
If you find our work helpful or relevant to your research, please kindly cite our paper:
```text
@misc{fan2025makesgoodspeechtokenizer,
      title={What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study}, 
      author={Xiaoran Fan and Zhichao Sun and Yangfan Gao and Jingfei Xiong and Hang Yan and Yifei Cao and Jiajun Sun and Shuo Li and Zhihao Zhang and Zhiheng Xi and Yuhao Zhou and Senjie Jin and Changhao Jiang and Junjie Ye and Ming Zhang and Rui Zheng and Zhenhua Han and Yunke Zhang and Demei Yan and Shaokang Dong and Tao Ji and Tao Gui and Qi Zhang and Xuanjing Huang},
      year={2025},
      eprint={2506.12537},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.12537}, 
}
```
