import os
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import json


print(torch.cuda.is_available())
print(torch.version.cuda)
# Create the "data" folder in the parent directory if it doesn't exist
os.makedirs("../data", exist_ok=True)


# Load the dataset and specify the cache directory
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
# dataset = load_dataset("Trelis/chess_pieces", cache_dir="../data")




print(dataset["test"][0])

# print(json.dumps(dataset["test"][1]["messages"], indent=2))

# print(json.dumps(dataset["test"][0]["messages"], indent=2))
# message = dataset["test"][1]["messages"]
# print(message)
# nb_messages = len(message)//2
# print(nb_messages)
# prompt = message[2]["content"][0]["text"]
# print(prompt)

# response = message[3]["content"][0]["text"]
# print(response)

# image = dataset["test"][1]["images"][0]

# image.show()