import logging
import uuid
from functools import partial
from typing import List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


def collate_fn(examples: List[Dict[str, Any]], processor: PaliGemmaProcessor, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """
    Collate function that now assumes images have been preprocessed to RGB,
    reducing redundant work during training.
    """
    texts = ["<image>ocr\n" for _ in examples]
    labels = [example["label"] for example in examples]
    # Directly use the preprocessed image
    images = [example["image"] for example in examples]
    
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest"
    )
    return tokens.to(dtype)

def prepare_datasets(ds: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    """
    Prepare and split datasets for training, validation and testing.
    
    Args:
        ds: Raw dataset from hub
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Convert each row into two examples (one per augmented image)
    def split_augmented(example):
        results = []
        for img_type, img in enumerate((
            example.get("augmented_front_plate"),
            example.get("augmented_rear_plate")
        )):
            if img is not None:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                results.append({
                    "image": img, 
                    "label": example["vrn"],
                    "image_type": "front" if img_type == 0 else "rear"
                })
        return results

    # Create flattened dataset
    flattened_data = []
    for example in ds:
        flattened_data.extend(split_augmented(example))

    ds_aug = Dataset.from_dict({
        "image": [x["image"] for x in flattened_data],
        "label": [x["label"] for x in flattened_data]
    })

    # Split ratios
    total_examples = len(ds_aug)
    train_size = int(0.7 * total_examples)
    val_size = int(0.15 * total_examples)
    
    # Split dataset
    ds_aug = ds_aug.shuffle(seed=42)
    train_ds = ds_aug.select(range(0, train_size))
    val_ds = ds_aug.select(range(train_size, train_size + val_size))
    test_ds = ds_aug.select(range(train_size + val_size, total_examples))
    
    # Validate splits
    for name, dataset in [
        ("training", train_ds),
        ("validation", val_ds),
        ("test", test_ds)
    ]:
        logger.info(f"{name.capitalize()} dataset size: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError(f"{name} dataset is empty!")

    return train_ds, val_ds, test_ds

def main():
    # Configuration
    CONFIG = {
        "model_id": "google/paligemma2-3b-pt-448",
        "batch_size": 4,
        "num_epochs": 5,
        "learning_rate": 1e-5,
        "lora_rank": 8,
        "lora_dropout": 0.1,
        "output_dir": "Paligemma2-3B-448-UK-Car-VRN",
        "repo_name": "UK-Car-Plate-OCR-PaLiGemma"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare datasets
    ds = load_dataset("spawn99/UK-Car-Plate-VRN-Dataset", split="train")
    train_ds, val_ds, test_ds = prepare_datasets(ds)
    
    # Initialize model and processor
    processor = PaliGemmaProcessor.from_pretrained(CONFIG["model_id"])

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        CONFIG["model_id"], 
        device_map="auto", 
        quantization_config=bnb_config,
        attn_implementation="eager"
    )
    

    # Setup LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Adjust learning rate for LoRA
    LEARNING_RATE = CONFIG["learning_rate"]

    model.train()  
    DTYPE = model.dtype

    # 6. Setup the training arguments using a cosine scheduler.
    run_id = str(uuid.uuid4())[:8]
    
    # Calculate total training steps (using floor division)
    num_training_steps = (len(train_ds) // CONFIG["batch_size"]) * CONFIG["num_epochs"]
    warmup_steps = num_training_steps // 15

    # Calculate steps for saves and evaluations
    save_steps = num_training_steps // 10  # 10 saves total

    # Ensure save_steps is even to allow eval_steps = save_steps/2 exactly
    if save_steps % 2 != 0:
        save_steps += 1

    eval_steps = save_steps // 2  # Evaluate twice as frequently as saving
    
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Saving every {save_steps} steps")
    logger.info(f"Evaluating every {eval_steps} steps")

    training_args = TrainingArguments(
num_train_epochs=CONFIG["num_epochs"],
    remove_unused_columns=False,
    warmup_steps=warmup_steps,
    learning_rate=LEARNING_RATE,
    weight_decay=1e-5,
    logging_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=5,
    output_dir=CONFIG["output_dir"],
    max_grad_norm=1.0,
    bf16=True,
    report_to=["wandb"],
    run_name=f"paligemma-vrn-{run_id}",
    eval_strategy="steps",
    eval_steps=eval_steps,
    dataloader_pin_memory=False,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit", 
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    metric_for_best_model="sequence_accuracy",
    greater_is_better=True,
    per_device_train_batch_size=CONFIG["batch_size"],  
    per_device_eval_batch_size=CONFIG["batch_size"],   
    eval_accumulation_steps=4
    )

     # Define a custom compute_metrics function to track evaluation metrics
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_idsw
        decoded_predictions = [processor.decode(pred, skip_special_tokens=True).replace("ocr\n", "").strip() for pred in predictions]
        decoded_labels = [processor.decode(label, skip_special_tokens=True).strip() for label in labels]
        accuracy = sum(p == l for p, l in zip(decoded_predictions, decoded_labels)) / len(decoded_predictions)
        return {
            "sequence_accuracy": accuracy
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
        repo_name=CONFIG["repo_name"],  # Choose your desired repository name
        commit_message=f"Best model checkpoint - Accuracy: {results.metrics['accuracy']:.4f}",
        blocking=True  # Wait until the upload is complete
    )

    logger.info("Model successfully pushed to Hub!")

if __name__ == '__main__':
    main() 