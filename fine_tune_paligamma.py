import torch
from datasets import load_dataset, Dataset
import os
# Set OpenMP threads before other imports
os.environ["OMP_NUM_THREADS"] = "1"

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    logging
)
from PIL import Image
import uuid
import datetime

# Set a global verbosity level (INFO, DEBUG, etc.)
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# Removed the custom trainer with layer-specific learning rates.
# We now rely on the default optimizer creation (AdamW) from Trainer.

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
    tokens = tokens.to(dtype)
    return tokens

class CollateWrapper:
    def __init__(self, processor, device, dtype):
        self.processor = processor
        self.device = device
        self.dtype = dtype
    
    def __call__(self, examples):
        return collate_fn(examples, self.processor, self.device, self.dtype)

def main():
    # Initialize process group with better error handling
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank != -1:
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(minutes=15)
            )
            torch.cuda.set_device(local_rank)
            logger.info(f"Initialized process group for rank {local_rank} of {world_size}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {str(e)}")
            raise

    # Move BATCH_SIZE definition to top of main()
    BATCH_SIZE = 4
    
    # Adjust device assignment
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else "cpu"
    
    # Scale batch size by world size
    global_batch_size = BATCH_SIZE * world_size
    
    # Only print on main process
    if local_rank <= 0:
        logger.info(f"Running with {world_size} GPUs")
        logger.info(f"Global batch size: {global_batch_size}")
    
    # Memory and CUDA optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')  # Enable TF32 for better performance

    # Adjust batch size for multi-GPU setup
    num_epochs = 20
    gradient_accumulation_steps = 1
    num_workers = min(4, os.cpu_count()//2)  # Workers per GPU

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
    logger.info(f"Total available examples: {total_available}")
    
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
    
    # Use standard cosine scheduler instead of the custom scheduler.
    warmup_steps = total_training_steps // num_epochs  # Approx. 5% of total steps for warmup

    logger.info("\nTraining Configuration:")
    logger.info(f"Total batches per epoch: {batches_per_epoch}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_training_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    ds_aug = ds_aug.shuffle(seed=42)
    ds_aug = ds_aug.select(range(total_examples))
    
    # Log the total number of examples selected.
    logger.info(f"Total examples selected: {total_examples}")

    # Split the dataset.
    train_ds = ds_aug.select(range(0, train_count))
    val_ds = ds_aug.select(range(train_count, train_count + val_count))
    test_ds = ds_aug.select(range(train_count + val_count, total_examples))
    
    # Assertions and logging to verify dataset splits.
    assert len(train_ds) == train_count, f"Expected {train_count} train examples, got {len(train_ds)}"
    assert len(val_ds) == val_count, f"Expected {val_count} validation examples, got {len(val_ds)}"
    assert len(test_ds) == test_count, f"Expected {test_count} test examples, got {len(test_ds)}"
    
    logger.info(f"Train dataset size: {len(train_ds)}")
    logger.info(f"Validation dataset size: {len(val_ds)}")
    logger.info(f"Test dataset size: {len(test_ds)}")
    
    # Pre-process images before training
    logger.info("Pre-processing training images...")
    train_ds = train_ds.map(
        lambda x: {"image": x["image"].convert("RGB") if isinstance(x["image"], str) else x["image"]}, 
        load_from_cache_file=False
    )
    logger.info("Pre-processing validation images...")
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

    # Create dummy input for warmup pass
    dummy_text = ["<image>ocr\n"]
    dummy_image = Image.new('RGB', (448, 448))  # Create blank image of correct size
    dummy_input = processor(
        text=dummy_text,
        images=[dummy_image],
        return_tensors="pt",
        padding="longest"
    ).to(device)

    # Warmup pass to initialize CUDA
    logger.info("Performing CUDA warmup pass...")
    with torch.no_grad():
        dummy_output = model.generate(**dummy_input, max_new_tokens=20)
        torch.cuda.synchronize()  # Add explicit sync
    del dummy_input, dummy_output
    torch.cuda.empty_cache()

    model.train()  # Unfreeze entire model.
    DTYPE = model.dtype

    # 6. Setup the training arguments using a cosine scheduler.
    run_id = str(uuid.uuid4())[:8]

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
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
        dataloader_pin_memory=True,
        lr_scheduler_type="cosine",  # Use standard cosine scheduler.
        dataloader_num_workers=num_workers,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        # Multi-GPU specific arguments:
        local_rank=-1,
        parallel_mode="distributed",
    )

    # 7. Initialize the Trainer with training and evaluation datasets.
    trainer = Trainer(
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
    logger.info(f"Test results: {results.metrics}")

    # 10. Optionally, push your model to the Hugging Face Hub.
    trainer.push_to_hub()


if __name__ == '__main__':
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        main()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        # Make sure to clean up distributed process group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise 