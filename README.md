# What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study
[![arXiv](https://img.shields.io/badge/arXiv-2506.12537-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.12537)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://cnxupupup.github.io/SLM-Decoupled-MTP-Demo)


Official implementation of paper **Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction**.

## Overview
Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the role of speech tokenizer designs in LLM-centric SLMs, augmented by speech heads and speaker modeling. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12× faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.

## Citation
If you find our work helpful or relevant to your research, please kindly cite our paper:
```text
@misc{fan2025speechlanguagemodelsdecoupledtokenizers,
      title={Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction}, 
      author={Xiaoran Fan and Zhichao Sun and Yangfan Gao and Jingfei Xiong and Hang Yan and Yifei Cao and Jiajun Sun and Shuo Li and Zhihao Zhang and Zhiheng Xi and Yuhao Zhou and Senjie Jin and Changhao Jiang and Junjie Ye and Ming Zhang and Rui Zheng and Zhenhua Han and Yunke Zhang and Demei Yan and Shaokang Dong and Tao Ji and Tao Gui and Qi Zhang and Xuanjing Huang},
      year={2025},
      eprint={2506.12537},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.12537}, 
}
```
