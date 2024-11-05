import torch
import json
import numpy as np

from datasets import load_dataset
from transformers import (
    LlavaNextProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu


def main():

    model_id = "llava-hf/llama3-llava-next-8b-hf"
    dataset_id = "HuggingFaceH4/llava-instruct-mix-vsft"
    # dataset_id = "Trelis/chess_pieces"
    output_dir = f"../../model/{model_id.split('/')[1]}/{dataset_id.split('/')[1]}"
    logging_dir = "../../log"
    eval_batch_size = 2
    max_length = 4096

    processor = LlavaNextProcessor.from_pretrained(model_id)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
    )

    # apply_liger_kernel_to_llama()

    model_name = model_id.split("/")[1]

    with open(f"../model_config/{model_name}.json", "r") as f:
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

    val_dataset = load_dataset(dataset_id, split="test")

    def eval_collate_fn(examples):
        # inputs = []
        texts = []
        images = []
        for example in examples:
            # conversations = []
            nb_messages = len(example["messages"]) // 2
            for j in range(int(nb_messages)):
                length = 2 * j + 1
                conversations = example["messages"][:length]
                image = example["images"][0]
                # if j != 0:
                #     answer_text = example["messages"][2 * j - 1]["content"][0]["text"]
                #     conversations.append(
                #         {
                #             "role": "agent",
                #             "content": [
                #                 {"type": "text", "text": answer_text},
                #                 {"type": "image"},
                #             ],
                #         }
                #     )
                # if example["messages"][2 * j]["content"][0]["index"] == 0:
                #     input_text = example["messages"][2 * j]["content"][1]["text"]
                # else:
                #     input_text = example["messages"][2 * j]["content"][0]["text"]
                # image = example["images"][0]
                # conversations.append(
                #     {
                #         "role": "user",
                #         "content": [
                #             {"type": "text", "text": input_text},
                #             {"type": "image"},
                #         ],
                #     }
                # )
                # conversations = [
                #     {
                #     "role": "user",
                #     "content": [
                #         {"type": "text", "text": input_text},
                #         {"type": "image"},
                #     ],
                #     }
                # ]

                    
                

                text = processor.apply_chat_template(conversations, tokenize=False)
                texts.append(text)
                images.append(image)
        # images = [example["images"] for example in examples]

        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        
        # labels = batch["input_ids"].clone()
        # labels[labels == processor.tokenizer.pad_token_id] = -100
        # batch["labels"] = labels
        
        # assert all(
        #     value is not None for value in batch.values()
        # ), "Batch contains None values"
        
        return batch


    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        # BLEU
        bleu_scores = [sentence_bleu([ref], pred) for pred, ref in zip(decoded_preds, decoded_labels)]
        
        
        bertscore_result = bert_score.compute(predictions=decoded_preds, references=decoded_labels, lang="en")

        is_correct = [pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]
        return {
            "bleu": bleu_scores["bleu"],
            "bertscore_precision": np.mean(bertscore_result["precision"]),
            "bertscore_recall": np.mean(bertscore_result["recall"]),
            "bertscore_f1": np.mean(bertscore_result["f1"]),
            "correct_predictions": sum(is_correct),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        per_device_eval_batch_size=eval_batch_size,
        logging_dir=logging_dir,
        log_level="info",
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        data_collator=eval_collate_fn,
        compute_metrics=compute_metrics,
    )

    test_results = trainer.predict(val_dataset)
    print(test_results)
    # Extract and save test results to a JSON file
    output_json_path = f"{output_dir}/test_results.json"

    # Format data to be JSON serializable
    results_data = {
        "predictions": test_results.predictions.tolist() if isinstance(test_results.predictions, np.ndarray) else test_results.predictions,
        "metrics": test_results.metrics,
    }

    # Save results to a JSON file
    with open(output_json_path, "w") as f:
        json.dump(results_data, f, indent=4)


if __name__ == "__main__":
    main()
