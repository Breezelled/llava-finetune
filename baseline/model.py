from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from data_inspection import dataset
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
from pycocoevalcap.cider.cider import Cider

print(torch.cuda.is_available())
print(torch.version.cuda)


def display_image(image):
    if isinstance(image, Image.Image):
        # If the image is a PIL Image, display it directly
        image.show()
    else:
        # If the image is in array format, use matplotlib
        plt.imshow(image)
        plt.axis("off")  # Hide the axes for a cleaner look
        plt.show()


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

processor.patch_size = patch_size
processor.vision_feature_select_strategy = vision_feature_select_strategy


# Placeholder accuracy and BLEU score trackers
correct_predictions = 0
total_predictions = 0
bleu_scores = []
cider_scores = []
bert_scores_p, bert_scores_r, bert_scores_f1 = [], [], []
results = []

batch_size = 4
batch_inputs = []
batch_images = []
batch_expected_outputs = []

cider_scorer = Cider()

# Evaluate on a subset of the dataset (for example, 100 samples)
for i, example in enumerate(
    tqdm(dataset["test"], desc="Processing examples")
):  # Change the slice as needed
    print(f"Example {i+1}/{len(dataset['test'])}")
    nb_messages = len(example["messages"]) // 2
    for j in range(int(nb_messages)):
        print(f"Message {j+1}/{nb_messages}")
        if example["messages"][2 * j]["content"][0]["index"] == 0:
            input_text = example["messages"][2 * j]["content"][1][
                "text"
            ]  # Adjust based on the dataset format
        else:
            input_text = example["messages"][2 * j]["content"][0][
                "text"
            ]  # Adjust based on the dataset format
        image = example["images"][0]  # Adjust based on the dataset format

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image"},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(
            model.device
        )

        expected_output = example["messages"][2 * j + 1]["content"][0][
            "text"
        ]  # Adjust based on the dataset format

        # Get the model's output (can limit to one sample if your model generates multiple responses)
        output = model.generate(
            **inputs, max_new_tokens=128
        )  # Adjust if model outputs multiple fields
        original_text = processor.decode(output[0], skip_special_tokens=True)
        response_split = original_text.split("\n\n\n")[2:]
        model_output = " ".join(response_split)
        print(f"Expected Output: {expected_output}")
        print(f"Model Output: {model_output}")
        # For accuracy (if it's a classification task, modify this part if needed)
        is_correct = model_output.strip() == expected_output.strip()
        if model_output.strip() == expected_output.strip():
            correct_predictions += 1

        # For BLEU score (assuming expected_output is reference and model_output is the hypothesis)
        reference = [expected_output.split()]  # Reference sentence (tokenized)
        hypothesis = model_output.split()  # Hypothesis sentence (tokenized)

        bleu = sentence_bleu(reference, hypothesis)
        print(f"BLEU Score: {bleu:.4f}")
        bleu_scores.append(bleu)
        
        # CIDEr
        # decoded_labels = [expected_output]
        # decoded_preds = [model_output]
        # cider_score, _ = cider_scorer.compute_score(expected_output, model_output)
        # cider_scores.append(cider_score)
        # print(f"CIDEr Score: {cider_score:.4f}")
        
        # BERT Score
        # P, R, F1 = bert_score([model_output], [expected_output], lang="en")
        # bert_scores_p.append(P.item())
        # bert_scores_r.append(R.item())
        # bert_scores_f1.append(F1.item())
        # print(f"BERT Score: P={P.item():.4f}, R={R.item():.4f}, F1={F1.item():.4f}")

        total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions

# Calculate average BLEU score
average_bleu = np.mean(bleu_scores)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Average CIDEr Score: {np.mean(cider_scores):.4f}")
print(f"Average BERT Score: P={np.mean(bert_scores_p):.4f}, R={np.mean(bert_scores_r):.4f}, F1={np.mean(bert_scores_f1):.4f}")

