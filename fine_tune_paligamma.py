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
    
    # Basic configurations
    BATCH_SIZE = 2
    num_epochs = 20
    gradient_accumulation_steps = 1

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

    # 4. Load the model and processor
    model_id = "google/paligemma2-3b-pt-448"
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to(device)


    model.train()  # Unfreeze entire model.
    DTYPE = model.dtype

    # 6. Setup the training arguments using a cosine scheduler.
    run_id = str(uuid.uuid4())[:8]
    
    # Calculate warmup steps (typically 10% of total steps)
    num_training_steps = (len(train_ds) // (BATCH_SIZE * gradient_accumulation_steps)) * num_epochs
    warmup_steps = num_training_steps // 10

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=1e-5,  
        weight_decay=0.01,
        logging_steps=200,    
        save_strategy="epoch",  
        save_total_limit=3,
        output_dir="Paligemma2-3B-448-UK-Car-VRN",
        max_grad_norm=1.0,
        bf16=True,
        report_to=["wandb"],
        run_name=f"paligemma-vrn-{run_id}",
        eval_strategy="steps",
        eval_steps=200,      # More frequent evaluation
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        group_by_length=False,
    )

    # 7. Initialize the Trainer with a custom compute_loss method
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, processor=processor, dtype=DTYPE),
        args=training_args,
    )

    # 8. Launch training.
    trainer.train()

    # 9. Evaluate on the test dataset.
    results = trainer.predict(test_ds)
    logger.info(f"Test results: {results.metrics}")

    # 10. Optionally, push your model to the Hugging Face Hub.
    trainer.push_to_hub()


if __name__ == '__main__':
    main() 