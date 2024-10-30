from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from data_inspection import dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import json
import gc


print(torch.cuda.is_available())
print(torch.version.cuda)


def display_image(image):
    if isinstance(image, Image.Image):
        # If the image is a PIL Image, display it directly
        image.show()
    else:
        # If the image is in array format, use matplotlib
        plt.imshow(image)
        plt.axis('off')  # Hide the axes for a cleaner look
        plt.show()


# Create the "model" folder in the parent directory if it doesn't exist
os.makedirs("../model", exist_ok=True)

# Specify the cache directory
cache_dir = "../model"
model_id = "llava-hf/llama3-llava-next-8b-hf"

processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir=cache_dir)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    attn_implementation="sdpa",
    cache_dir=cache_dir
    )

with open(f"../model_config/llama3-llava-next-8b-hf.json", "r") as f:
    config = json.load(f)
    
patch_size = config["vision_config"]["patch_size"]
vision_feature_select_strategy = config["vision_feature_select_strategy"]

processor.patch_size = patch_size
processor.vision_feature_select_strategy = vision_feature_select_strategy

# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)

# display_image(image)

# # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image") 
# test_prompt = "What is shown in this image?"

# conversation = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "What is shown in this image?"},
#             {"type": "image"},
#         ],
#     },
# ]
# # Expanding inputs for image tokens in LLaVa-NeXT should be done in processing. Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. Using processors without these attributes in the config is deprecated and will throw an error in v4.47.
# # Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=128)
# # print(processor.decode(output[0], skip_special_tokens=True))
# original_text = processor.decode(output[0], skip_special_tokens=True)
# print(json.dump(original_text, f, indent=4))
# model_output = '\n'.join(original_text.split('\n')[2:]).strip()
# print(model_output)

# Placeholder accuracy and BLEU score trackers
correct_predictions = 0
total_predictions = 0
bleu_scores = []

# Evaluate on a subset of the dataset (for example, 100 samples)
for i, example in enumerate(dataset['test']):  # Change the slice as needed
    print(f"Example {i+1}/{len(dataset['test'])}")
    nb_messages = len(example['messages']) // 2
    for j in range(int(nb_messages)):
        print(f"Message {j+1}/{nb_messages}")
        if example['messages'][2*j]["content"][0]["index"] == 0:
            input_text = example['messages'][2*j]["content"][1]["text"]  # Adjust based on the dataset format
        else:
            input_text = example['messages'][2*j]["content"][0]["text"]  # Adjust based on the dataset format
        print(type(input_text))
        image = example['images'][0]  # Adjust based on the dataset format
        
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
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        expected_output = example['messages'][2*j+1]["content"][0]["text"]  # Adjust based on the dataset format

        # Get the model's output (can limit to one sample if your model generates multiple responses)
        output = model.generate(**inputs, max_new_tokens=128)  # Adjust if model outputs multiple fields
        original_text = processor.decode(output[0], skip_special_tokens=True) 
        response_split = original_text.split("\n\n\n")[2:]
        model_output = " ".join(response_split)
        # For accuracy (if it's a classification task, modify this part if needed)
        if model_output.strip() == expected_output.strip():
            correct_predictions += 1

        # For BLEU score (assuming expected_output is reference and model_output is the hypothesis)
        reference = [expected_output.split()]  # Reference sentence (tokenized)
        hypothesis = model_output.split()  # Hypothesis sentence (tokenized)
        
        bleu = sentence_bleu(reference, hypothesis)
        print(f"BLEU Score: {bleu:.4f}")
        bleu_scores.append(bleu)

        total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions

# Calculate average BLEU score
average_bleu = np.mean(bleu_scores)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu:.4f}")

