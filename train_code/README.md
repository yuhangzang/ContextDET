# ContextDET

This repo contains the official implement of "Contextual Object Detection with Multimodal Large Language Models".

# Introduction

ContextDET is a contextual object detection framework that leverages large language models (e.g., GPT-2, OPT) to power object detection with the language understanding, contextual understanding, open-vocabulary, and interactive feedback abilities.

Some of the code is based on the pure python implementation of [Deformable DETR][deformable-detr].

# Setting Up Environment

``` bash
conda create -n contextdet
conda activate contextdet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip install transformers
```
