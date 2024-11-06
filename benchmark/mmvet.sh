#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file LLaVA/playground/data/eval/mm-vet/mm-vet.jsonl \
    --image-folder LLaVA/playground/data/eval/mm-vet/images \
    --answers-file LLaVA/playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p LLaVA/playground/data/eval/mm-vet/results

python LLaVA/scripts/convert_mmvet_for_eval.py \
    --src LLaVA/playground/data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --dst LLaVA/playground/data/eval/mm-vet/results/llava-v1.5-7b.json

