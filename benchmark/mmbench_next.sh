#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path lmms-lab/llama3-llava-next-8b \
    --question-file LLaVA/playground/data/eval/mmbench_next/$SPLIT.tsv \
    --answers-file LLaVA/playground/data/eval/mmbench_next/answers/$SPLIT/llama3-llava-next-8b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p LLaVA/playground/data/eval/mmbench_next/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file LLaVA/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir LLaVA/playground/data/eval/mmbench_next/answers/$SPLIT \
    --upload-dir LLaVA/playground/data/eval/mmbench_next/answers_upload/$SPLIT \
    --experiment llama3-llava-next-8b
