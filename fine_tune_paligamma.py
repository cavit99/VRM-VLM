import torch
from functools import partial

from datasets import load_dataset, Dataset

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    logging
)
from PIL import Image
import uuid


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def collate_fn(examples, processor, dtype):
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
    tokens = tokens.to(dtype)
    return tokens

def main():
    # Simplified device setup for single GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjusted configurations for better GPU utilization
    BATCH_SIZE = 6  # Increased from 4 to utilize more GPU memory
    num_epochs = 10
    gradient_accumulation_steps = 1  # Reduced since we increased batch size

    # 1. Load the dataset from the Hub.
    # The dataset was previously created with create_dataset.py.
    # It has 7500 rows with augmented images in the columns:
    # "augmented_front_plate" and "augmented_rear_plate", and the registration number in "vrn".
    ds = load_dataset("spawn99/UK-Car-Plate-VRN-Dataset", split="train")

    # 2. Cast the image columns to Image type. This decodes them into PIL Images.
    # Instead of directly using the Image class, use a function to create images
    def create_pil_image(img_data):
        if isinstance(img_data, str):
            return Image.open(img_data)
        return img_data

    ds = ds.map(
        lambda x: {
            "front_plate": create_pil_image(x["front_plate"]),
            "rear_plate": create_pil_image(x["rear_plate"]),
            "augmented_front_plate": create_pil_image(x["augmented_front_plate"]),
            "augmented_rear_plate": create_pil_image(x["augmented_rear_plate"])
        }
    )

    # 3. Convert each row into two examples (one per augmented image).
    def split_augmented(example):
        results = []
        for img_type, img in enumerate((
            example.get("augmented_front_plate"),
            example.get("augmented_rear_plate")
        )):
            if img is not None:
                # Basic validation
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                results.append({
                    "image": img, 
                    "label": example["vrn"],
                    "image_type": "front" if img_type == 0 else "rear"
                })
        return results

    # Create flattened dataset directly
    flattened_data = []
    for example in ds:
        flattened_data.extend(split_augmented(example))

    # Add data validation after creating the dataset
    def validate_dataset(dataset, name="dataset"):
        """Validate dataset integrity"""
        if len(dataset) == 0:
            raise ValueError(f"{name} is empty!")
        logger.info(f"Dataset {name}: {len(dataset)} samples")
        return True

    # After creating datasets, add validation:
    ds_aug = Dataset.from_dict({
        "image": [x["image"] for x in flattened_data],
        "label": [x["label"] for x in flattened_data]
    })

    # Validate main dataset
    validate_dataset(ds_aug, "main dataset")

    # After flattening, we have approximately 15,000 examples
    total_available = len(ds_aug)
    logger.info(f"Total available examples: {total_available}")
    
    # Simplify batch size calculations
    total_examples = len(ds_aug)
    train_size = int(0.7 * total_examples)
    val_size = int(0.15 * total_examples)
    test_size = total_examples - train_size - val_size

    # Split the dataset with shuffling
    ds_aug = ds_aug.shuffle(seed=42) 
    
    # Split the dataset directly
    train_ds = ds_aug.select(range(0, train_size))
    val_ds = ds_aug.select(range(train_size, train_size + val_size))
    test_ds = ds_aug.select(range(train_size + val_size, total_examples))
    
    # Validate each subset
    validate_dataset(train_ds, "training dataset")
    validate_dataset(val_ds, "validation dataset")
    validate_dataset(test_ds, "test dataset")
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(val_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")

    # 4. Load the model and processor with eager attention
    model_id = "google/paligemma2-3b-pt-448"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'  # Added eager attention
    ).to(device)


    model.train()  # Unfreeze entire model.
    DTYPE = model.dtype

    # 6. Setup the training arguments using a cosine scheduler.
    run_id = str(uuid.uuid4())[:8]
    
    # Calculate total training steps (using floor division)
    num_training_steps = (len(train_ds) // (BATCH_SIZE * gradient_accumulation_steps)) * num_epochs
    warmup_steps = num_training_steps // 15

    # Calculate steps for evenly spaced saves and evaluations
    save_steps = num_training_steps // 10  # 10 saves total
    eval_steps = save_steps  # Evaluate at the same frequency as saving
    
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Saving and evaluating every {save_steps} steps")

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=eval_steps,     # More frequent logging
        save_strategy="steps",
        save_steps=save_steps,        # 10 saves total
        save_total_limit=10,
        output_dir="Paligemma2-3B-448-UK-Car-VRN",
        max_grad_norm=1.0,
        bf16=True,
        report_to=["wandb"],
        run_name=f"paligemma-vrn-{run_id}",
        eval_strategy="steps",
        eval_steps=eval_steps,        # More frequent evaluation
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        fp16_full_eval=True,
        dataloader_num_workers=4,
        torch_compile=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Define a custom compute_metrics function to track evaluation metrics
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        return {
            "accuracy": (predictions == labels).mean()
        }

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, processor=processor, dtype=DTYPE),
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # 8. Launch training.
    trainer.train()

    # 9. Evaluate on the test dataset.
    results = trainer.predict(test_ds)
    logger.info(f"Test results: {results.metrics}")

    # 10. Get the best checkpoint path and push to Hub
    best_ckpt_path = trainer.state.best_model_checkpoint
    logger.info(f"Best checkpoint path: {best_ckpt_path}")
    
    # Push the best model to the Hub
    trainer.push_to_hub(
        repo_name="UK-Car-Plate-OCR-PaLiGemma",  # Choose your desired repository name
        commit_message=f"Best model checkpoint - Accuracy: {results.metrics['accuracy']:.4f}",
        blocking=True  # Wait until the upload is complete
    )

    logger.info("Model successfully pushed to Hub!")

if __name__ == '__main__':
    main() 