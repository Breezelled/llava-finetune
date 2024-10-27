from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from data_inspection import dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os

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
processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf", cache_dir=cache_dir)
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir) 

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
test_prompt = "What is shown in this image?"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": test_prompt},
            {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))



# Test on a few samples from the dataset



# # Encode image + text and pass them to the model (requires vision support)
# # vision_model = SomeVisionModel(...)
# # inputs = vision_model(image, text_input)
# # outputs = model.generate(inputs)
# # Placeholder accuracy and BLEU score trackers
# correct_predictions = 0
# total_predictions = 0
# bleu_scores = []

# # Evaluate on a subset of the dataset (for example, 100 samples)
# for i, example in enumerate(dataset['test']):  # Change the slice as needed
#     input_text = example['input']  # Adjust based on the dataset format
#     expected_output = example['output']  # Adjust based on the dataset format

#     # Get the model's output (can limit to one sample if your model generates multiple responses)
#     model_output = model(input_text)[0]['generated_text']  # Adjust if model outputs multiple fields

#     # For accuracy (if it's a classification task, modify this part if needed)
#     if model_output.strip() == expected_output.strip():
#         correct_predictions += 1

#     # For BLEU score (assuming expected_output is reference and model_output is the hypothesis)
#     reference = [expected_output.split()]  # Reference sentence (tokenized)
#     hypothesis = model_output.split()  # Hypothesis sentence (tokenized)
#     bleu = sentence_bleu(reference, hypothesis)
#     bleu_scores.append(bleu)

#     total_predictions += 1

# # Calculate accuracy
# accuracy = correct_predictions / total_predictions

# # Calculate average BLEU score
# average_bleu = np.mean(bleu_scores)

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Average BLEU Score: {average_bleu:.4f}")

 # Display the image
