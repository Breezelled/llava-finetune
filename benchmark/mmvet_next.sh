#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path lmms-lab/llama3-llava-next-8b \
    --question-file LLaVA/playground/data/eval/mm-vet_next/mm-vet.jsonl \
    --image-folder LLaVA/playground/data/eval/mm-vet_next/images \
    --answers-file LLaVA/playground/data/eval/mm-vet_next/answers/llama3-llava-next-8b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p LLaVA/playground/data/eval/mm-vet_next/results

python LLaVA/scripts/convert_mmvet_for_eval.py \
    --src LLaVA/playground/data/eval/mm-vet_next/answers/llama3-llava-next-8b.jsonl \
    --dst LLaVA/playground/data/eval/mm-vet_next/results/llama3-llava-next-8b.json

