#!/usr/bin/bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
accelerate config
ccelerate launch fine_tune_wsj.py  --data_dir ./wsj_style2 --pretrained_model_name_or_path ./models/stable-diffusion-v1-4/