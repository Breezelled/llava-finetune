import torch
import json

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
from torch.optim import AdamW
from liger_kernel.transformers import apply_liger_kernel_to_llama



def main():

    model_id = "llava-hf/llama3-llava-next-8b-hf"
    dataset_id = "Trelis/chess_pieces"
    output_dir = f"../model/{model_id.split('/')[1]}/{dataset_id.split('/')[1]}-1epoch-better-prompt"
    train_batch_size = 1
    gradient_accumulation_steps = 1
    num_train_epochs = 1
    learning_rate = 2e-5
    weight_decay = 0.0
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"
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
        attn_implementation="sdpa",
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

    def find_all_linear_names(model):
        cls = torch.nn.Linear
        lora_module_names = set()
        multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
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
        r=2,
        lora_alpha=4,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        # target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config).to("cuda")

    train_dataset = load_dataset(dataset_id, split="train")

    def train_collate_fn(examples):
        texts = []
        for example in examples:
            image = example["image"]  # Adjust based on the dataset format

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the chess pieces in the image, including their types and colors."},
                        {"type": "image"},
                    ],
                },
            ]
            texts.append(processor.apply_chat_template(conversation, tokenize=False))
        images = [example["image"] for example in examples]

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

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        assert all(
            value is not None for value in batch.values()
        ), "Batch contains None values"

        return batch

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        save_strategy="steps",
        # save_steps=30000,
        per_device_train_batch_size=train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=1,
        logging_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        bf16=True,
        push_to_hub=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        remove_unused_columns=False,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, fused=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=train_collate_fn,
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(output_dir)

    # test_results = trainer.predict(val_dataset)
    # print(test_results)


if __name__ == "__main__":
    main()