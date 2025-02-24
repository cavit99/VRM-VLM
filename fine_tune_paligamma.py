import torch
from datasets import load_dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from PIL import Image

# Define a custom Trainer that applies layer-specific learning rates.
class CustomTrainer(Trainer):
    def create_optimizer(self):
        # Build the optimizer with three parameter groups.
        optimizer = torch.optim.AdamW([
            # Vision encoder: very low LR to gently adjust visual features.
            {"params": self.model.vision_tower.parameters(), "lr": 5e-6},
            # Multi-modal projector: moderate LR for alignment tuning.
            {"params": self.model.multi_modal_projector.parameters(), "lr": 2e-5},
            # LLM head: higher LR for refining text generation.
            {"params": self.model.language_model.parameters(), "lr": 5e-5},
        ], max_grad_norm=1.0)  # Add gradient clipping to prevent exploding gradients
        return optimizer

def main():
    # 1. Load the dataset from the Hub.
    # The dataset was previously created with create_dataset.py.
    # It has 7500 rows with augmented images in the columns:
    # "augmented_front_plate" and "augmented_rear_plate", and the registration number in "vrn".
    ds = load_dataset("spawn99/UK-Car-Plate-VRN-Dataset", split="train")
    
    # 2. Convert each row into two examples (one per augmented image).
    # Each example will contain a field "image" (the augmented image) and "label" (the VRN).
    def split_augmented(example):
        examples = []
        # Check if an augmented image exists (it should on both sides)
        if example.get("augmented_front_plate") is not None:
            examples.append({
                "image": example["augmented_front_plate"],
                "label": example["vrn"]
            })
        if example.get("augmented_rear_plate") is not None:
            examples.append({
                "image": example["augmented_rear_plate"],
                "label": example["vrn"]
            })
        return examples

    # Use map and flatten instead of explode
    ds_aug = ds.map(
        split_augmented,
        remove_columns=ds.column_names,
        batched=False
    ).flatten()
    # After flat_map, we have approximately 15,000 examples.
    # For perfect divisibility by 32, we adjust the splits as follows:
    train_count = 312 * 32   # 312 batches = 9,984 examples for training.
    val_count   = 62 * 32    # 62 batches = 1,984 examples for validation.
    test_count  = 62 * 32    # 62 batches = 1,984 examples for testing.
    total_examples = train_count + val_count + test_count  # = 13,952 examples

    ds_aug = ds_aug.shuffle(seed=42)
    ds_aug = ds_aug.select(range(total_examples))
    train_ds = ds_aug.select(range(0, train_count))
    val_ds = ds_aug.select(range(train_count, train_count + val_count))
    test_ds = ds_aug.select(range(train_count + val_count, total_examples))
    
    # 4. Load the model and processor
    model_id = "google/paligemma2-3b-pt-448"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16  # or use 'torch.float16' if bf16 isn't available.
    ).to(device)
    
    model.train()  # Unfreeze entire model.
    DTYPE = model.dtype

    # 5. Define a collate function.
    def collate_fn(examples):
        texts = ["<image>ocr\n" for _ in examples]
        labels = [example["label"] for example in examples]
        images = []
        for example in examples:
            img = example["image"]
            if isinstance(img, str):
                img = Image.open(img)
            images.append(img.convert("RGB"))
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest"
        )
        tokens = tokens.to(DTYPE).to(device)
        return tokens

    # 6. Setup the training arguments.
    training_args = TrainingArguments(
        num_train_epochs=20,          
        remove_unused_columns=False,
        per_device_train_batch_size=32, 
        warmup_steps=300,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        output_dir="Paligemma2-3B-448-UK-Car-VRN",
        max_grad_norm=1.0,            
        label_smoothing_factor=0.1,    
        bf16=True,
        report_to=["wandb"],
        run_name="paligemma-vrn-finetune",  # Add a descriptive run name
        evaluation_strategy="steps",    # Add evaluation strategy
        eval_steps=500,                # Add evaluation frequency
        dataloader_pin_memory=False,
        lr_scheduler_type="warmup_stable_decay", 
    )

    # 7. Initialize the custom Trainer with training and evaluation datasets.
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=training_args,
    )

    # 8. Launch training.
    trainer.train()

    # 9. Evaluate on the test dataset.
    results = trainer.predict(test_ds)
    print("Test results:", results.metrics)

    # 10. Optionally, push your model to the Hugging Face Hub.
    trainer.push_to_hub()


if __name__ == '__main__':
    main() 