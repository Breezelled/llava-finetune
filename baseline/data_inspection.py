import os
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import json


print(torch.cuda.is_available())
print(torch.version.cuda)
# Create the "data" folder in the parent directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# Specify the cache directory
cache_dir = "../data"

# Load the dataset and specify the cache directory
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", cache_dir=cache_dir)


# Display a small portion of the dataset

# Display the first example in the dataset
print(dataset["test"][0])

# Obtain the prompt and the response from the first example
message = dataset["test"][0]["messages"]

prompt = message[0]["content"][1]["text"]
print(prompt)

response = message[1]["content"][0]["text"]
print(response)

image = dataset["test"][0]["images"][0]

image.show()