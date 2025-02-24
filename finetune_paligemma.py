import numpy as np
import torch
import uuid
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import BitsAndBytesConfig

# Set device to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and output configurations
MODEL_ID = "google/paligemma2-3b-pt-448"
OUTPUT_DIR = "Paligemma2-3B-448-UK-Car-VRN"
DATASET_NAME = "spawn99/UK-Car-Plate-VRN-Dataset"
HUB_MODEL_ID = "spawn99/Paligemma2-3B-448-UK-Car-VRN"

def load_and_prepare_dataset(subset_ratio=1.0):
    """
    Loads and flattens the UK Car Plate VRN dataset from Hugging Face for augmented images only.
    Optionally sub-sample the dataset to a subset for quick testing.
    
    Parameters:
        subset_ratio (float): The fraction of the dataset to use (default is 1.0 for 100%).
    
    Returns:
        train_ds, valid_ds, test_ds: The training, validation, and test dataset splits.
    """
    # Load the raw dataset (assumed to be on the Hub)
    raw_ds = load_dataset(DATASET_NAME, split="train")
    
    # Only consider augmented image fields
    samples = []
    image_fields = ["augmented_front_plate", "augmented_rear_plate"]
    for record in raw_ds:
        for field in image_fields:
            if record[field] is not None:
                samples.append({
                    "image": record[field],
                    "prompt": "ocr",  # our task is OCR
                    "target": record["vrn"]
                })
    
    # Create a new Hugging Face Dataset from these samples and shuffle it
    ds = Dataset.from_list(samples)
    ds = ds.shuffle(seed=42)
    
    # Optionally use a subset of the dataset for testing purposes
    if subset_ratio < 1.0:
        subset_size = int(len(ds) * subset_ratio)
        ds = ds.select(range(subset_size))
    
    # Split the dataset into train (70%), validation (20%), and test (10%)
    ds_train_temp = ds.train_test_split(test_size=0.3, seed=42)
    train_ds = ds_train_temp["train"]
    temp_ds = ds_train_temp["test"]
    
    # Out of the remaining 30%, use 2/3 for validation (~20%) and 1/3 for test (~10%)
    temp_split = temp_ds.train_test_split(test_size=0.3333, seed=42)
    valid_ds = temp_split["train"]
    test_ds = temp_split["test"]
    
    return train_ds, valid_ds, test_ds

def collate_fn(batch):
    # Batch contains dictionaries with "image", "prompt", and "target" keys
    images = [item["image"] for item in batch]
    prefixes = [f"<image>{item['prompt']}" for item in batch]  # Add the image token to the prompt
    suffixes = [item["target"] for item in batch]  # Use the target as the suffix

    # Process the inputs using the processor
    inputs = processor(
        text=prefixes,
        images=images,
        return_tensors="pt",
        suffix=suffixes,
        padding="longest"
    )
    
    # Do not move the data to GPU here.
    return inputs

def compute_metrics(eval_preds):
    """
    Computes exact-match accuracy for OCR.
    
    It decodes both predictions and labels and then checks how many predictions match the target exactly.
    """
    preds, labels = eval_preds
    # Replace any -100 (ignore index) with the pad token id for proper decoding.
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute exact match accuracy
    exact_matches = [pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]
    accuracy = np.mean(exact_matches)
    return {"exact_match_accuracy": accuracy}

def main():
    global processor
    
    # Clear CUDA cache before starting to ensure maximum available memory
    torch.cuda.empty_cache()
    
    # -------------------------------------------------------
    # Load the PaLiGemma processor.
    # -------------------------------------------------------
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    
    # -------------------------------------------------------
    # Configure 4-bit quantization via BitsAndBytes.
    # -------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # -------------------------------------------------------
    # Load the pre-trained PaLiGemma model using 4-bit quantization.
    # -------------------------------------------------------
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    
    # -------------------------------------------------------
    # Set up LoRA for efficient fine-tuning.
    # -------------------------------------------------------
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # -------------------------------------------------------
    # Freeze the vision encoder layers to reduce the number of trainable parameters.
    # -------------------------------------------------------
    if hasattr(model, "vision_tower"):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    if hasattr(model, "multi_modal_projector"):
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False
    
    model.to(DEVICE)
    
    # -------------------------------------------------------
    # Prepare the dataset with an even smaller subset for evaluation
    # -------------------------------------------------------
    train_ds, valid_ds, test_ds = load_and_prepare_dataset(subset_ratio=0.1)
    
    # Use a very small validation dataset to prevent OOM
    valid_ds = valid_ds.select(range(min(50, len(valid_ds))))
    test_ds = test_ds.select(range(min(50, len(test_ds))))
    
    # -------------------------------------------------------
    # Define training hyperparameters.
    # -------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=20,
        save_steps=20,
        logging_steps=100,
        learning_rate=5e-5,
        optim="adamw_bnb_8bit",
        weight_decay=1e-6,
        max_grad_norm=1.0,
        adam_beta2=0.999,
        warmup_steps=5,
        run_name=f"paligemma-vrn-{str(uuid.uuid4())[:8]}",
        save_total_limit=1,
        remove_unused_columns=False,
        bf16=True,  # Changed from bf16=True to ensure consistent precision
        label_names=["labels"],
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        eval_accumulation_steps=4,    # Process evaluation predictions in smaller chunks
    )
    
    # -------------------------------------------------------
    # Initialize the Trainer.
    # -------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    
    # -------------------------------------------------------
    # Start training.
    # -------------------------------------------------------
    trainer.train()
    
    # -------------------------------------------------------
    # For final evaluation, do it in smaller batches with controlled memory usage
    # -------------------------------------------------------
    print("Final evaluation on validation set:")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        eval_results = trainer.evaluate(max_length=20, num_beams=1)
    print("Validation Results:", eval_results)
    
    print("Final evaluation on test set:")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        test_results = trainer.evaluate(test_ds, max_length=20, num_beams=1)
    print("Test Results:", test_results)
    
    # -------------------------------------------------------
    # Save the final trained model and processor.
    # -------------------------------------------------------
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    # -------------------------------------------------------
    # Push the final model and processor to the Hugging Face Hub.
    # -------------------------------------------------------
    #model.push_to_hub(HUB_MODEL_ID, use_auth_token=True)
    #processor.push_to_hub(HUB_MODEL_ID, use_auth_token=True)
    #print(f"Model and processor pushed to: https://huggingface.co/{HUB_MODEL_ID}")

if __name__ == "__main__":
    main()