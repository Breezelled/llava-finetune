import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
)
import evaluate
from tqdm import tqdm
from PIL import Image

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    dataset_id = "HuggingFaceH4/llava-instruct-mix-vsft"
    max_length = 4096
    max_new_tokens = 256

    # Initialize processor and model
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Set image processing parameters
    processor.image_processor.size = {"height": 336, "width": 336}
    # Ensure processor uses model's config
    processor.image_processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = getattr(
        model.config, 'vision_feature_select_strategy', 'default'
    )
    processor.patch_size = processor.image_processor.patch_size

    # Load 20% of the test dataset
    dataset = load_dataset(dataset_id, split="test")

    # Define the batch size
    batch_size = 1  

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=lambda x: x, num_workers=4
    )

    # Initialize metrics
    correct_predictions = 0
    total_predictions = 0
    bleu_scores = []
    bertscore_scores = {"precision": [], "recall": [], "f1": []}

    # Initialize outputs
    ground_truths = []
    generated_outputs = []

    # Load evaluation metrics
    bleu_metric = evaluate.load("bleu")
    bertscore_metric = evaluate.load("bertscore")

    # Create dummy image
    def create_dummy_image():
        return Image.new('RGB', (336, 336), color='white')

    # Iterate over the DataLoader to evaluate the model in batches
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Process each example in the batch
        for example in batch:
            image_list = example["images"]  # List of images in the example
            conversation = example["messages"]
            conversation_context = []  # Initialize conversation context for this example
            image_index = 0  # To track the current image index

            # Iterate over messages in the conversation
            for message in conversation:
                role = message["role"]
                content = message["content"]
                content_items = []

                images = []  # Reset images for each message

                for item in content:
                    if item["type"] == "text":
                        content_items.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        content_items.append({"type": "image"})
                        # Ensure we have enough images
                        if image_index < len(image_list):
                            images.append(image_list[image_index])
                            image_index += 1
                        else:
                            raise ValueError("Image index out of range")
                    else:
                        pass  # Handle other types if necessary

                conversation_context.append({"role": role, "content": content_items})

                # If the role is 'assistant', we need to generate a response
                if role == "assistant":
                    # Prepare the prompt up to this point
                    prompt_context = conversation_context[:-1]  # Exclude the assistant's reply
                    prompt = processor.apply_chat_template(prompt_context, add_generation_prompt=True)

                    # Prepare inputs for the model
                    if images:
                        # Use provided images
                        inputs = processor(
                            images=images, text=prompt, return_tensors="pt", padding=True, max_length=max_length
                        ).to(model.device)
                    else:
                        # Use dummy image
                        dummy_image = create_dummy_image()
                        inputs = processor(
                            images=[dummy_image], text=prompt, return_tensors="pt", padding=True, max_length=max_length
                        ).to(model.device)

                    # Ensure inputs are in the correct dtype
                    inputs = inputs.to(torch.bfloat16)

                    # Generate response from the model
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    model_output = processor.decode(output_ids[0], skip_special_tokens=True)

                    # Extract the ground truth answer
                    ground_truth = ''.join([item['text'] for item in message['content'] if item['type'] == 'text']).strip()

                    # Append results for evaluation
                    ground_truths.append(ground_truth)
                    generated_outputs.append(model_output.strip())

                    # Increment total prediction count
                    total_predictions += 1

                    # Calculate accuracy (exact match)
                    if model_output.strip() == ground_truth:
                        correct_predictions += 1

                    # Update conversation context with the model's output
                    conversation_context.append({"role": "assistant", "content": [{"type": "text", "text": model_output.strip()}]})

    # Evaluate metrics after processing all batches
    if generated_outputs:
        # Calculate metrics
        bleu_result = bleu_metric.compute(predictions=generated_outputs, references=[[gt] for gt in ground_truths])
        average_bleu = bleu_result["bleu"]

        bertscore_result = bertscore_metric.compute(predictions=generated_outputs, references=ground_truths, lang="en")
        average_bertscore = {
            "precision": np.mean(bertscore_result["precision"]),
            "recall": np.mean(bertscore_result["recall"]),
            "f1": np.mean(bertscore_result["f1"]),
        }

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Print final evaluation results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Average BLEU Score: {average_bleu:.4f}")
        print(f"Average BERTScore Precision: {average_bertscore['precision']:.4f}")
        print(f"Average BERTScore Recall: {average_bertscore['recall']:.4f}")
        print(f"Average BERTScore F1: {average_bertscore['f1']:.4f}")
    else:
        print("No outputs generated for evaluation.")

if __name__ == "__main__":
    main()
