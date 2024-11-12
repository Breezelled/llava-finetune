import torch
import json
import re
import os
import numpy as np
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def main(model_id, is_few_shot=False):
    pretrained_id = "llava-hf/llama3-llava-next-8b-hf"
    dataset_id = "Trelis/chess_pieces"
    eval_batch_size = 1
    max_length = 4096
    max_new_tokens = 256
    prompt = "Describe the chess pieces in the image, including their types and colors."

    if is_few_shot:
        prompt = "Given 5 examples: \
        Question: Describe the chess pieces in the image, including their types and colors. Answer: 'A white knight.', \
        Question: Describe the chess pieces in the image, including their types and colors. Answer: 'A black pawn and a black rook.', \
        Question: Describe the chess pieces in the image, including their types and colors. Answer: 'A white pawn, a black queen and a white knight. Portions of other chess pieces are also visible.', \
        Question: Describe the chess pieces in the image, including their types and colors. Answer: 'A white king, a white queen and a black king - all standing close together near a chess board.', \
        Question: Describe the chess pieces in the image, including their types and colors. Answer: 'A white bishop with the bottom of some black chess pieces in the background.' \
        Now, describe the chess pieces in the image, including their types and colors."

    processor = LlavaNextProcessor.from_pretrained(pretrained_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    model_name = pretrained_id.split("/")[1]
    with open(f"../model_config/{model_name}.json", "r") as f:
        config = json.load(f)

    patch_size = config["vision_config"]["patch_size"]
    vision_feature_select_strategy = config["vision_feature_select_strategy"]
    image_size = config["vision_config"]["image_size"]
    processor.patch_size = patch_size
    processor.vision_feature_select_strategy = vision_feature_select_strategy
    processor.image_processor.size = {
        "height": image_size,
        "width": image_size,
    }

    test_dataset = load_dataset(dataset_id, split="test")

    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, collate_fn=lambda x: x, num_workers=4
    )

    bleu_metric = evaluate.load("bleu")
    bertscore_metric = evaluate.load("bertscore")

    def evaluate_fn(prompt_text):
        total_predictions = 0
        correct_predictions = 0
        bleu_scores = []
        bertscore_scores = {"precision": [], "recall": [], "f1": []}
        results = []

        for batch in tqdm(test_dataloader, desc="Processing examples"):
            for example in batch:

                image = example["image"]
                expected_output = example["caption"]

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    model.device
                )

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                model_output = processor.decode(output_ids[0], skip_special_tokens=True)

                match = re.search(
                    r"assistant\s*(.*)", model_output, re.IGNORECASE | re.DOTALL
                )
                model_output = match.group(1).strip() if match else model_output.strip()

                ground_truths = [expected_output]
                generated_outputs = [model_output]

                print(f"Ground Truth: {expected_output}")
                print(f"Model Output: {model_output}")

                if model_output.strip() == expected_output.strip():
                    correct_predictions += 1

                total_predictions += 1

                bleu_result = bleu_metric.compute(
                    predictions=generated_outputs,
                    references=[[gt] for gt in ground_truths],
                )
                bleu_scores.append(bleu_result["bleu"])

                bertscore_result = bertscore_metric.compute(
                    predictions=generated_outputs,
                    references=ground_truths,
                    lang="en",
                    model_type="roberta-large",
                )
                bertscore_scores["precision"].extend(bertscore_result["precision"])
                bertscore_scores["recall"].extend(bertscore_result["recall"])
                bertscore_scores["f1"].extend(bertscore_result["f1"])

                results.append({
                    "Ground Truth": expected_output,
                    "Model Output": model_output,
                    "BLEU Score": bleu_result["bleu"],
                    "BERT Score": {
                        "precision": bertscore_result["precision"][0],
                        "recall": bertscore_result["recall"][0],
                        "f1": bertscore_result["f1"][0],
                    },
                    "Correct Prediction": model_output.strip() == expected_output.strip(),
                })

                print(
                    f"Batch BLEU: {bleu_result['bleu']:.4f}, Batch BERTScore F1: {np.mean(bertscore_result['f1']):.4f}"
                )

        accuracy = correct_predictions / total_predictions
        average_bleu = np.mean(bleu_scores)
        average_bertscore = {
            "precision": np.mean(bertscore_scores["precision"]),
            "recall": np.mean(bertscore_scores["recall"]),
            "f1": np.mean(bertscore_scores["f1"]),
        }

        output_data = {
            "results": results,
            "summary": {
                "Accuracy": accuracy,
                "Average BLEU Score": average_bleu,
                "Average BERTScore": average_bertscore,
            }
        }

        os.makedirs("./results", exist_ok=True)
        shot_num = 5 if is_few_shot else 0
        output_file = f"./results/{model_id.split('/')[-1]}_{shot_num}shot_results.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Average BLEU Score: {average_bleu:.4f}")
        print(f"Average BERTScore Precision: {average_bertscore['precision']:.4f}")
        print(f"Average BERTScore Recall: {average_bertscore['recall']:.4f}")
        print(f"Average BERTScore F1: {average_bertscore['f1']:.4f}")

    evaluate_fn(prompt)


if __name__ == "__main__":
    finetuned_general_model_id = (
        "../model/llama3-llava-next-8b-hf/llava-instruct-mix-vsft"
    )
    finetuned_specific_1_model_id = (
        "../model/llama3-llava-next-8b-hf/chess_pieces-1epoch-better-prompt"
    )
    finetuned_specific_3_model_id = (
        "../model/llama3-llava-next-8b-hf/chess_pieces-3epoch-better-prompt"
    )
    pretrained_model_id = "llava-hf/llama3-llava-next-8b-hf"

    # zeroshot
    main(finetuned_general_model_id)
    main(finetuned_specific_1_model_id)
    # main(finetuned_specific_3_model_id)
    main(pretrained_model_id)

    # 5-shot
    main(finetuned_general_model_id, is_few_shot=True)
    main(finetuned_specific_1_model_id, is_few_shot=True)
    # main(finetuned_specific_3_model_id, is_few_shot=True)
    main(pretrained_model_id, is_few_shot=True)
