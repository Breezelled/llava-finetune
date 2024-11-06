#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file LLaVA/playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT

python LLaVA/scripts/convert_mmbench_for_submission.py \
    --annotation-file LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir LLaVA/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir LLaVA/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
