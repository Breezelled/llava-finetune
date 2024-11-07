from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig, AutoConfig
import torch
from PIL import Image
import requests
import json
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import evaluate

# Check if GPU is available
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model ID and cache directory for data
model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
# cache_dir = "../data"
# os.makedirs(cache_dir, exist_ok=True)

# torch.backends.cuda.matmul.allow_tf32 = True
# print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

# Load the LLaVA-OneVision model and processor
processor = AutoProcessor.from_pretrained(model_id)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    #     load_in_4bit=True
    #     quantization_config=quantization_config,
)

# Manually convert the bias to bfloat16
# for name, param in model.named_parameters():
#     if "bias" in name:
#         param.data = param.data.type(torch.bfloat16)


# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)

# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "What is shown in this image?"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# # inputs["pixel_values"] = inputs["pixel_values"].type(torch.float16)
# inputs = inputs.to(torch.bfloat16)

# output = model.generate(**inputs, max_new_tokens=10)
# print(processor.decode(output[0], skip_special_tokens=True))


# Initialize counters and metrics for evaluation
correct_predictions = 0
total_predictions = 0
bleu_scores = []
bertscore_scores = {"precision": [], "recall": [], "f1": []}

# Load the dataset for evaluation
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="test")

# Define the batch size
batch_size = 1

# Create DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x)

bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")

# Iterate over the DataLoader to evaluate the model in batches
for batch in tqdm(dataloader, desc="Processing batches"):
    images = []
    conversation_contexts = [[] for _ in range(len(batch))]  # Initialize conversation context for each example in the batch
    ground_truths = []
    generated_outputs = []

    # Process each example in the batch
    for i, example in enumerate(batch):
        image = example["images"][0]
        nb_messages = len(example["messages"]) // 2

        for j in range(int(nb_messages)):
            if example["messages"][2 * j]["content"][0]["index"] == 0:
                input_text = example["messages"][2 * j]["content"][1]["text"]
            else:
                input_text = example["messages"][2 * j]["content"][0]["text"]

            ground_truth = example["messages"][2 * j + 1]["content"][0]["text"]  # Assistant's true answer

            # Accumulate context by adding current input and model output from previous rounds
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

            # Prepare input for model generation
            images.append(image)
            input_texts = [prompt]

            # Prepare inputs for the model
            inputs = processor(images=images, text=input_texts, return_tensors="pt", padding=True).to(model.device)
            inputs = inputs.to(torch.bfloat16)

            # Generate responses from the model
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=128)
            model_output = processor.decode(output_ids[0], skip_special_tokens=True)

            # Update conversation contexts with model output and ground truth for metrics
            conversation_contexts[i].append({"role": "user", "content": [{"type": "text", "text": input_text}]})
            conversation_contexts[i].append({"role": "assistant", "content": [{"type": "text", "text": model_output}]})
            ground_truths.append(ground_truth)
            generated_outputs.append(model_output)

            # Accuracy (exact match)
            if model_output.strip() == ground_truth.strip():
                correct_predictions += 1

            # BLEU score
            # reference = [ground_truth.split()]
            # hypothesis = model_output.split()
            # bleu = sentence_bleu(reference, hypothesis)
            # bleu_scores.append(bleu)

            # Clear the images and prompts for the next round in the conversation
            images.clear()
            input_texts.clear()

    # Increment total prediction count
    total_predictions += len(ground_truths)

    bleu_result = bleu_metric.compute(predictions=generated_outputs, references=[[gt] for gt in ground_truths])
    bleu_scores.append(bleu_result["bleu"])

    bertscore_result = bertscore_metric.compute(predictions=generated_outputs, references=ground_truths, lang="en", model_type="roberta-large")
    bertscore_scores["precision"].extend(bertscore_result["precision"])
    bertscore_scores["recall"].extend(bertscore_result["recall"])
    bertscore_scores["f1"].extend(bertscore_result["f1"])
    print(f"BLEU: {bleu_result['bleu']:.4f}, BERTScore F1: {np.mean(bertscore_result['f1']):.4f}")

# Calculate final accuracy and average BLEU score across the dataset
accuracy = correct_predictions / total_predictions
average_bleu = np.mean(bleu_scores)
average_bertscore = {
    "precision": np.mean(bertscore_scores["precision"]),
    "recall": np.mean(bertscore_scores["recall"]),
    "f1": np.mean(bertscore_scores["f1"]),
}

# Print final evaluation results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Average BERTScore Precision: {average_bertscore['precision']:.4f}")
print(f"Average BERTScore Recall: {average_bertscore['recall']:.4f}")
print(f"Average BERTScore F1: {average_bertscore['f1']:.4f}")


#
# # Load the dataset for evaluation
# dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="test", cache_dir=cache_dir)
#
# # Iterate over the dataset to evaluate the model
# for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
#     print(f"Example {i + 1}/{len(dataset)}")
#
#     # Extract image and text prompt from the dataset
#     image = example["images"][0]  # Get the image from the example
#     nb_messages = len(example["messages"]) // 2
#
#     for j in range(int(nb_messages)):
#         print(f"Message {j + 1}/{nb_messages}")
#
#         if example["messages"][2 * j]["content"][0]["index"] == 0:
#             input_text = example["messages"][2 * j]["content"][1]["text"]
#         else:
#             input_text = example["messages"][2 * j]["content"][0]["text"]
#
#         expected_output = example["messages"][2 * j + 1]["content"][0]["text"]  # Expected response text
#         conversation = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": input_text},
#                     {"type": "image"},
#                 ],
#             },
#         ]
#         prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#
#         # Prepare inputs for model generation
#         inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
#         inputs = inputs.to(torch.bfloat16)
#
#         # Generate response from the model
#         output = model.generate(**inputs, max_new_tokens=128)
#         model_output = processor.decode(output[0], skip_special_tokens=True)
#
#         # Print the prompt, model output, and expected output for comparison
#         print("Prompt:", prompt)
#         print("Model Output:", model_output)
#         print("Expected Output:", expected_output)
#
#         # Calculate accuracy (exact match comparison)
#         if model_output.strip() == expected_output.strip():
#             correct_predictions += 1
#
#         # Calculate BLEU score
#         reference = [expected_output.split()]  # Tokenized reference text
#         hypothesis = model_output.split()  # Tokenized model output
#         bleu = sentence_bleu(reference, hypothesis)
#         print(f"BLEU Score: {bleu:.4f}")
#         bleu_scores.append(bleu)
#
#         # Increment total prediction count
#         total_predictions += 1
#
# # Calculate final accuracy and average BLEU score across the dataset
# accuracy = correct_predictions / total_predictions
# average_bleu = np.mean(bleu_scores)
#
# # Print final evaluation results
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Average BLEU Score: {average_bleu:.4f}")
