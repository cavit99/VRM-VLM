import torch
import multiprocessing
from datasets import load_dataset, Dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from PIL import Image
import os
import uuid

# Define a custom Trainer that applies layer-specific learning rates.
class CustomTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.vision_tower.parameters(), "lr": 2.5e-6},
            {"params": self.model.multi_modal_projector.parameters(), "lr": 1e-5},
            {"params": self.model.language_model.parameters(), "lr": 2.5e-5},
        ])

# Move both collate functions outside of main() to make them pickleable
def collate_fn(examples, processor, device, dtype):
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
    tokens = tokens.to(dtype).to(device)
    return tokens

class CollateWrapper:
    def __init__(self, processor, device, dtype):
        self.processor = processor
        self.device = device
        self.dtype = dtype
    
    def __call__(self, examples):
        return collate_fn(examples, self.processor, self.device, self.dtype)

def main():
    # Memory and CUDA optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')  # Enable TF32 for better performance

    # Reduce batch size and number of workers to help with memory issues
    BATCH_SIZE = 4  # Reduced from 8
    num_epochs = 20
    gradient_accumulation_steps = 2  # Increased to maintain effective batch size
    num_workers = 2  # Reduced number of workers

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

    # After flattening, we have approximately 15,000 examples
    total_available = len(ds_aug)
    print(f"Total available examples: {total_available}")
    
    # Calculate the largest number divisible by both BATCH_SIZE and our split ratio (70/15/15)
    batches_per_epoch = (total_available // BATCH_SIZE // num_epochs) * num_epochs  # Round down to nearest multiple of epochs
    total_examples = batches_per_epoch * BATCH_SIZE
    
    # Calculate split sizes (70% train, 15% val, 15% test)
    train_count = (batches_per_epoch * 14 // num_epochs) * BATCH_SIZE  # 70%
    val_count = (batches_per_epoch * 3 // num_epochs) * BATCH_SIZE    # 15%
    test_count = (batches_per_epoch * 3 // num_epochs) * BATCH_SIZE   # 15%
    
    # Training steps calculation
    
    effective_batch_size = BATCH_SIZE * gradient_accumulation_steps
    steps_per_epoch = train_count // effective_batch_size
    total_training_steps = steps_per_epoch * num_epochs
    
    # Log all calculations for verification
    print("\nTraining Configuration:")
    print(f"Total batches per epoch: {batches_per_epoch}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")

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
    
    # Pre-process images before training
    print("Pre-processing training images...")
    train_ds = train_ds.map(
        lambda x: {"image": x["image"].convert("RGB") if isinstance(x["image"], str) else x["image"]}, 
        load_from_cache_file=False
    )
    print("Pre-processing validation images...")
    val_ds = val_ds.map(
        lambda x: {"image": x["image"].convert("RGB") if isinstance(x["image"], str) else x["image"]}, 
        load_from_cache_file=False
    )
    test_ds = test_ds.map(
        lambda x: {"image": x["image"].convert("RGB") if isinstance(x["image"], str) else x["image"]}, 
        load_from_cache_file=False
    )

    # 4. Load the model and processor
    model_id = "google/paligemma2-3b-pt-448"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Warmup pass to initialize CUDA
    print("Performing CUDA warmup pass...")
    dummy_input = processor(
        text=["<image>ocr\n"], 
        images=[Image.new("RGB", (448,448))],
        return_tensors="pt"
    ).to(device)
    model.generate(**dummy_input, max_new_tokens=20)

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
    # Compute steps based on dataset size, batch size, and gradient accumulation
    effective_batch_size = BATCH_SIZE * gradient_accumulation_steps
    steps_per_epoch = len(train_ds) // effective_batch_size
    total_training_steps = steps_per_epoch * num_epochs
    
    # Calculate scheduler steps
    warmup_steps = total_training_steps // num_epochs  # 5% of total steps for warmup
    decay_steps = total_training_steps // 10   # 10% of total steps for decay
    stable_steps = total_training_steps - warmup_steps - decay_steps  # Remaining steps
    
    print(f"Training steps calculation:")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Decay steps: {decay_steps}")
    print(f"Stable steps: {stable_steps}")
    # Generate a short UUID (first 8 characters)
    run_id = str(uuid.uuid4())[:8]

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,  
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
        run_name=f"paligemma-vrn-{run_id}",
        eval_strategy="steps",
        eval_steps=500,
        dataloader_pin_memory=False,  # Disabled pin_memory to reduce memory usage
        lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={
            "num_decay_steps": decay_steps,
            "num_stable_steps": stable_steps,
            "min_lr_ratio": 0.1
        },
        dataloader_num_workers=num_workers,  # Reduced number of workers
    )

    # 7. Initialize the custom Trainer with training and evaluation datasets.
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=CollateWrapper(processor, device, DTYPE),
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
    multiprocessing.set_start_method('spawn', force=True)
    main() 