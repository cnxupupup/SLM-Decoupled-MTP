# Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction
[![arXiv]()]()
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://cnxupupup.github.io/SLM-Decoupled-MTP-Demo)


Official implementation of paper **Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction**.

## Overview
Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross- modal alignment and high-quality speech generation. In this work, we systematically investigate the impact of key components (i.e., speech tokenizers, speech heads, and speaker modeling) on the performance of LLM-centric SLMs. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode
multiple speech tokens. This leads to up to **12× faster** decoding and a substantial **drop in Word Error Rate (from 6.07 to 3.01)**. Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.


## Citation
```text

```