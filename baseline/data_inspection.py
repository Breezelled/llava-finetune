import os
from datasets import load_dataset

# Create the "data" folder in the parent directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# Specify the cache directory
cache_dir = "../data"

# Load the dataset and specify the cache directory
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", cache_dir=cache_dir)

# Display a small portion of the dataset
print(dataset)