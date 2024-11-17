import numpy as np
from data_inspection import dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader

max_length = 4096
max_new_tokens = 256

# Check if GPU is available
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)


# Create the "model" folder in the parent directory if it doesn't exist
# os.makedirs("../model", exist_ok=True)

# Specify the cache directory
# cache_dir = "../model"
model_id = "llava-hf/llama3-llava-next-8b-hf"

# torch.backends.cuda.matmul.allow_tf32 = True
print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

# Create the "data" folder in the parent directory if it doesn't exist
# os.makedirs("../data", exist_ok=True)


# Load the dataset and specify the cache directory
dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft", split="test")

processor = LlavaNextProcessor.from_pretrained(model_id)
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

# Placeholder accuracy and BLEU score trackers
correct_predictions = 0
total_predictions = 0
bleu_scores = []
bertscore_scores = {"precision": [], "recall": [], "f1": []}

batch_size = 1

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x, num_workers=4)

bleu_metric = evaluate.load("bleu")
bertscore_metric = evaluate.load("bertscore")

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
            inputs = processor(images=images, text=input_texts, return_tensors="pt", padding=True, max_length=max_length).to(model.device)
            inputs = inputs.to(torch.bfloat16)

            # Generate responses from the model
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
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
    print(f"BLEU: {bleu_result['bleu']:.4f}, BERTScore: {bertscore_result['f1']}")

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

# # Evaluate on a subset of the dataset (for example, 100 samples)
# for i, example in enumerate(
#     tqdm(dataset["test"], desc="Processing examples")
# ):
#     nb_messages = len(example["messages"]) // 2
#     for j in range(int(nb_messages)):
#         if example["messages"][2 * j]["content"][0]["index"] == 0:
#             input_text = example["messages"][2 * j]["content"][1]["text"]
#         else:
#             input_text = example["messages"][2 * j]["content"][0]["text"]
#         image = example["images"][0]
#         expected_output = example["messages"][2 * j + 1]["content"][0]["text"]

#         # Append data to batch lists
#         batch_inputs.append(input_text)
#         batch_images.append(image)
#         batch_expected_outputs.append(expected_output)
        
#         # Process when batch size is reached
#         if len(batch_inputs) == batch_size:
#             conversations = [
#                 [
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": batch_inputs[k]},
#                             {"type": "image"},
#                         ],
#                     }
#                 ]
#                 for k in range(batch_size)
#             ]
#             print(batch_images)
#             print(conversations)
#             prompts = [processor.apply_chat_template(conv, tokenize=False) for conv in conversations]
#             inputs = processor(images=batch_images, text=prompts, return_tensors="pt", padding=True,truncation=True,padding_side='left').to(model.device)

#             # Generate outputs
#             outputs = model.generate(**inputs, max_new_tokens=4096)

#             for k in range(batch_size):
#                 original_text = processor.decode(outputs[k], skip_special_tokens=True)
#                 print(original_text)
#                 response_split = original_text.split("\n\n\n")[2:]
#                 model_output = " ".join(response_split)
#                 expected_output = batch_expected_outputs[k]

#                 # For accuracy
#                 is_correct = model_output.strip() == expected_output.strip()
#                 if is_correct:
#                     correct_predictions += 1

#                 # For BLEU score
#                 reference = [expected_output.split()]
#                 hypothesis = model_output.split()
#                 bleu = sentence_bleu(reference, hypothesis)
#                 print(f"BLEU Score: {bleu:.4f}")
#                 bleu_scores.append(bleu)

#                 # # CIDEr score
#                 # decoded_labels = [expected_output]
#                 # decoded_preds = [model_output]
#                 # cider_score, _ = cider_scorer.compute_score(decoded_labels, decoded_preds)
#                 # cider_scores.append(cider_score)

#                 # BERT Score
#                 # P, R, F1 = bert_score([model_output], [expected_output], lang="en")
#                 # bert_scores_p.append(P.item())
#                 # bert_scores_r.append(R.item())
#                 # bert_scores_f1.append(F1.item())
#                 print(f"Expected Output: {expected_output}")
#                 print(f"Model Output: {model_output}")
                
                
#                 total_predictions += 1

#             # Clear batch lists
#             batch_inputs.clear()
#             batch_images.clear()
#             batch_expected_outputs.clear()

# # Calculate metrics
# accuracy = correct_predictions / total_predictions
# average_bleu = np.mean(bleu_scores)

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Average BLEU Score: {average_bleu:.4f}")
# # print(f"Average CIDEr Score: {np.mean(cider_scores):.4f}")
# # print(f"Average BERT Score: P={np.mean(bert_scores_p):.4f}, R={np.mean(bert_scores_r):.4f}, F1={np.mean(bert_scores_f1):.4f}")
