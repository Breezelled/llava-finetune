import torch
import json
import numpy as np
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def main():
    model_id = "../model/llama3-llava-next-8b-hf/llava-instruct-mix-vsft"
    pretrained_id = "llava-hf/llama3-llava-next-8b-hf"
    dataset_id = "HuggingFaceH4/llava-instruct-mix-vsft"
    eval_batch_size = 1
    max_length = 4096
    max_new_tokens = 256

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

    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=lambda x: x, num_workers=4)

    bleu_metric = evaluate.load("bleu")
    bertscore_metric = evaluate.load("bertscore")

    def evaluate_fn():
        total_predictions = 0
        correct_predictions = 0
        bleu_scores = []
        bertscore_scores = {"precision": [], "recall": [], "f1": []}

        for batch in tqdm(test_dataloader, desc="Processing batches"):
            images = []
            conversation_contexts = [[] for _ in range(len(batch))]
            ground_truths = []
            generated_outputs = []

            for i, example in enumerate(batch):
                image = example["images"][0]
                nb_messages = len(example["messages"]) // 2

                for j in range(nb_messages):
                    if example["messages"][2 * j]["content"][0]["index"] == 0:
                        input_text = example["messages"][2 * j]["content"][1]["text"]
                    else:
                        input_text = example["messages"][2 * j]["content"][0]["text"]

                    ground_truth = example["messages"][2 * j + 1]["content"][0]["text"]

                    conversation = conversation_contexts[i] + [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": input_text},
                                {"type": "image"},
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

                    images.append(image)
                    input_texts = [prompt]

                    inputs = processor(
                        images=images,
                        text=input_texts,
                        return_tensors="pt",
                        padding=True,
                        max_length=max_length,
                        truncation=True,
                    ).to("cuda")
                    inputs = inputs.to(torch.bfloat16)

                    with torch.no_grad():
                        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    model_output = processor.decode(output_ids[0], skip_special_tokens=True)

                    conversation_contexts[i].append({"role": "user", "content": [{"type": "text", "text": input_text}]})
                    conversation_contexts[i].append({"role": "assistant", "content": [{"type": "text", "text": model_output}]})
                    ground_truths.append(ground_truth)
                    generated_outputs.append(model_output)

                    if model_output.strip() == ground_truth.strip():
                        correct_predictions += 1

                    images.clear()
                    input_texts.clear()

            total_predictions += len(ground_truths)

            bleu_result = bleu_metric.compute(predictions=generated_outputs, references=[[gt] for gt in ground_truths])
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

            print(f"Batch BLEU: {bleu_result['bleu']:.4f}, Batch BERTScore F1: {np.mean(bertscore_result['f1']):.4f}")

        accuracy = correct_predictions / total_predictions
        average_bleu = np.mean(bleu_scores)
        average_bertscore = {
            "precision": np.mean(bertscore_scores["precision"]),
            "recall": np.mean(bertscore_scores["recall"]),
            "f1": np.mean(bertscore_scores["f1"]),
        }

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Average BLEU Score: {average_bleu:.4f}")
        print(f"Average BERTScore Precision: {average_bertscore['precision']:.4f}")
        print(f"Average BERTScore Recall: {average_bertscore['recall']:.4f}")
        print(f"Average BERTScore F1: {average_bertscore['f1']:.4f}")

    evaluate_fn()

if __name__ == "__main__":
    main()