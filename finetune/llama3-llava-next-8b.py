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
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from pycocoevalcap.cider.cider import Cider


def main():

    model_id = "llava-hf/llama3-llava-next-8b-hf"
    dataset_id = "HuggingFaceH4/llava-instruct-mix-vsft"
    output_dir = f"../../model/{model_id.split('/')[1]}/{dataset_id.split('/')[1]}"
    logging_dir = "../../log"
    train_batch_size = 4
    eval_batch_size = 2
    gradient_accumulation_steps = 1
    num_train_epochs = 1
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"

    processor = LlavaNextProcessor.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
    )

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

    print("Special tokens:", processor.tokenizer.special_tokens_map)
    print("Additional special tokens:", processor.tokenizer.additional_special_tokens)
    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["multi_modal_projector", "vision_model"]
        for name, module in model.named_modules():
            if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                continue
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        # target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    train_dataset = load_dataset(dataset_id, split="train")
    val_dataset = load_dataset(dataset_id, split="test")

    def train_collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        images = [example["images"] for example in examples]

        if isinstance(model, LlavaForConditionalGeneration):
            # LLava1.5 does not support multiple images
            images = [image[0] for image in images]

        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        )


        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        assert all(
            value is not None for value in batch.values()
        ), "Batch contains None values"

        return batch

    cider_scorer = Cider()

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        # BLEU
        bleu_scores = [sentence_bleu([ref], pred) for pred, ref in zip(decoded_preds, decoded_labels)]

        # CIDEr
        cider_scores, _ = cider_scorer.compute_score(decoded_labels, decoded_preds)

        # BERTScore
        P, R, F1 = bert_score(decoded_preds, decoded_labels, lang="en", rescale_with_baseline=True)

        return {
            "bleu": np.mean(bleu_scores),
            "cider": np.mean(cider_scores),
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=50000,
        save_strategy="steps",
        save_steps=50000,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=logging_dir,
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        bf16=True,
        push_to_hub=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_results = trainer.predict(val_dataset)
    print(test_results)


if __name__ == "__main__":
    main()
