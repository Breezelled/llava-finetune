from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import json
import gc
from tqdm import tqdm
from bert_score import score as bert_score
from datasets import load_dataset





print(torch.cuda.is_available())
print(torch.version.cuda)
# Create the "data" folder in the parent directory if it doesn't exist
os.makedirs("../data", exist_ok=True)


# Load the dataset and specify the cache directory
data_id = "Trelis/chess_pieces"
dataset = load_dataset("Trelis/chess_pieces", cache_dir="../data")

# Create the "model" folder in the parent directory if it doesn't exist
os.makedirs("../model", exist_ok=True)

# Specify the cache directory
cache_dir = "../model"
model_id = "llava-hf/llama3-llava-next-8b-hf"

torch.backends.cuda.matmul.allow_tf32 = True
print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

processor = LlavaNextProcessor.from_pretrained(model_id,cache_dir=cache_dir)
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    # quantization_config=quantization_config,
)

with open(f"../model_config/llama3-llava-next-8b-hf.json", "r") as f:
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

# Initialize variables
results = []
bleu_scores = []
bert_scores_p = []
bert_scores_r = []
bert_scores_f1 = []
correct_predictions = 0
total_predictions = 0

# Evaluate on a subset of the dataset (for example, 100 samples)
for i, example in enumerate(
    tqdm(dataset["test"], desc="Processing examples")
):  # Limit to 100 samples for example
    print(f"Example {i+1}/{len(dataset['test'])}")

    image = example["image"]  # Adjust based on the dataset format

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Caption this image."},
                {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(
        model.device
    )

    expected_output = example["caption"]  # Adjust based on the dataset format

    # Get the model's output
    output = model.generate(
        **inputs, max_new_tokens=128
    )  # Adjust if model outputs multiple fields
    original_text = processor.decode(output[0], skip_special_tokens=True)
    response_split = original_text.split("\n\n\n")[2:]
    model_output = " ".join(response_split)

    # Check accuracy
    is_correct = model_output.strip() == expected_output.strip()
    if is_correct:
        correct_predictions += 1

    # Calculate BLEU score
    reference = [expected_output.split()]  # Reference sentence (tokenized)
    hypothesis = model_output.split()  # Hypothesis sentence (tokenized)
    bleu = sentence_bleu(reference, hypothesis)
    bleu_scores.append(bleu)

    # Calculate BERT Score
    P, R, F1 = bert_score([model_output], [expected_output], lang="en", rescale_with_baseline=True)
    bert_scores_p.append(P.item())
    bert_scores_r.append(R.item())
    bert_scores_f1.append(F1.item())

    # Store result for this example
    results.append({
        "Example": i + 1,
        "Expected Output": expected_output,
        "Model Output": model_output,
        "BLEU Score": bleu,
        "BERT Score": {
            "P": P.item(),
            "R": R.item(),
            "F1": F1.item()
        },
        "Correct Prediction": is_correct
    })

    total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions

# Calculate average BLEU score
average_bleu = np.mean(bleu_scores)

# Store summary results
summary = {
    "Accuracy": accuracy,
    "Average BLEU Score": average_bleu,
    "Average BERT Score": {
        "P": np.mean(bert_scores_p),
        "R": np.mean(bert_scores_r),
        "F1": np.mean(bert_scores_f1)
    }
}

# Add summary to the results
output_data = {
    "results": results,
    "summary": summary
}

# Write the results to a JSON file
with open("./result/{model_id}_{data_id}_results.json", "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Average BERT Score: P={np.mean(bert_scores_p):.4f}, R={np.mean(bert_scores_r):.4f}, F1={np.mean(bert_scores_f1):.4f}")