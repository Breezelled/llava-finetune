from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from PIL import Image
import requests
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from datasets import load_dataset


model_name = "llava-next-8b"  # Replace with the actual model name if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

input_text = "Explain the process of phase transitions in neural networks."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



# Load the dataset
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")

# Display a small portion of the dataset
print(dataset)

# Test on a few samples from the dataset
for example in dataset['train'][:5]:
    input_text = example['input']  # Adjust depending on the dataset format
    expected_output = example['output']  # Adjust depending on the dataset format

    # Get the model's output
    model_output = model(input_text)

    # Compare the model's output with the expected output
    print(f"Input: {input_text}")
    print(f"Model Output: {model_output}")
    print(f"Expected Output: {expected_output}\n")


# Encode image + text and pass them to the model (requires vision support)
# vision_model = SomeVisionModel(...)
# inputs = vision_model(image, text_input)
# outputs = model.generate(inputs)
# Placeholder accuracy and BLEU score trackers
correct_predictions = 0
total_predictions = 0
bleu_scores = []

# Evaluate on a subset of the dataset (for example, 100 samples)
for i, example in enumerate(dataset['test']):  # Change the slice as needed
    input_text = example['input']  # Adjust based on the dataset format
    expected_output = example['output']  # Adjust based on the dataset format

    # Get the model's output (can limit to one sample if your model generates multiple responses)
    model_output = model(input_text)[0]['generated_text']  # Adjust if model outputs multiple fields

    # For accuracy (if it's a classification task, modify this part if needed)
    if model_output.strip() == expected_output.strip():
        correct_predictions += 1

    # For BLEU score (assuming expected_output is reference and model_output is the hypothesis)
    reference = [expected_output.split()]  # Reference sentence (tokenized)
    hypothesis = model_output.split()  # Hypothesis sentence (tokenized)
    bleu = sentence_bleu(reference, hypothesis)
    bleu_scores.append(bleu)

    total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions

# Calculate average BLEU score
average_bleu = np.mean(bleu_scores)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu:.4f}")

 