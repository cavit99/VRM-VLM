import torch
from datasets import load_dataset, Dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from PIL import Image
import os

# Define a custom Trainer that applies layer-specific learning rates.
class CustomTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.vision_tower.parameters(), "lr": 2.5e-6},
            {"params": self.model.multi_modal_projector.parameters(), "lr": 1e-5},
            {"params": self.model.language_model.parameters(), "lr": 2.5e-5},
        ])

def main():
    # Reduce batch size to help with memory issues
    BATCH_SIZE = 16  

    # Add memory management configuration
    torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some GPU memory free
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # 1. Load the dataset from the Hub.
    # The dataset was previously created with create_dataset.py.
    # It has 7500 rows with augmented images in the columns:
    # "augmented_front_plate" and "augmented_rear_plate", and the registration number in "vrn".
    ds = load_dataset("spawn99/UK-Car-Plate-VRN-Dataset", split="train")
    
    # 2. Convert each row into two examples (one per augmented image).
    def split_augmented(example):
        return [
            {"image": img, "label": example["vrn"]}
            for img in (
                example.get("augmented_front_plate"),
                example.get("augmented_rear_plate")
            )
            if img is not None
        ]

    # Create flattened dataset directly
    flattened_data = []
    for example in ds:
        flattened_data.extend(split_augmented(example))
    
    ds_aug = Dataset.from_dict({
        "image": [x["image"] for x in flattened_data],
        "label": [x["label"] for x in flattened_data]
    })

    # After flattening, we have approximately 15,000 examples.
    # For perfect divisibility by BATCH_SIZE, we adjust the splits as follows:
    train_count = 312 * BATCH_SIZE   # 312 batches = 9,984 examples for training
    val_count = 62 * BATCH_SIZE      # 62 batches = 1,984 examples for validation
    test_count = 62 * BATCH_SIZE     # 62 batches = 1,984 examples for testing
    total_examples = train_count + val_count + test_count  # = 13,952 examples

    ds_aug = ds_aug.shuffle(seed=42)
    ds_aug = ds_aug.select(range(total_examples))
    
    # Log the total number of examples selected.
    print(f"Total examples selected: {total_examples}")

    # Split the dataset.
    train_ds = ds_aug.select(range(0, train_count))
    val_ds = ds_aug.select(range(train_count, train_count + val_count))
    test_ds = ds_aug.select(range(train_count + val_count, total_examples))
    
    # Assertions and logging to verify dataset splits.
    assert len(train_ds) == train_count, f"Expected {train_count} train examples, got {len(train_ds)}"
    assert len(val_ds) == val_count, f"Expected {val_count} validation examples, got {len(val_ds)}"
    assert len(test_ds) == test_count, f"Expected {test_count} test examples, got {len(test_ds)}"
    
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
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

    # Compute total training steps.
    # Since there are 312 batches per epoch and we're training for 20 epochs:
    num_train_steps = 312 * 20  # = 6240 steps

    training_args = TrainingArguments(
        num_train_epochs=20,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,  # Added to maintain effective batch size
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
        run_name="paligemma-vrn-finetune",
        evaluation_strategy="steps",
        eval_steps=500,
        dataloader_pin_memory=False,
        lr_scheduler_type="warmup_stable_decay",
        # Pass additional scheduler arguments so that the scheduler has all required inputs.
        lr_scheduler_kwargs={
            "num_decay_steps": 624,
            "num_stable_steps": 5316,
            "min_lr_ratio": 0.1
       }
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