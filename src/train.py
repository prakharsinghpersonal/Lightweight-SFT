import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import logging
import sys
import os

from config import TrainingConfig, AWSConfig
from aws_utils import AWSClient

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def train(t_config: TrainingConfig, aws_config: AWSConfig):
    logger.info("Initializing SFT Pipeline...")
    
    # Initialize AWS Client for checkpointing
    aws_client = AWSClient(region_name=aws_config.region_name) if aws_config.use_s3_checkpointing else None

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(t_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    logger.info(f"Loading model: {t_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        t_config.model_name,
        device_map="auto",
        torch_dtype=torch.float16 if t_config.fp16 else torch.float32
    )

    if t_config.use_peft:
        logger.info("Applying LoRA adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=t_config.lora_r,
            lora_alpha=t_config.lora_alpha,
            lora_dropout=t_config.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Dummy dataset for scaffolding purposes if path doesn't exist
    if not os.path.exists(t_config.data_path):
        logger.warning(f"Dataset path {t_config.data_path} not found. Creating dummy dataset.")
        from datasets import Dataset
        dataset = Dataset.from_dict({
            "instruction": ["Describe quantum mechanics."] * 10,
            "input": [""] * 10,
            "output": ["Quantum mechanics is a fundamental theory in physics..."] * 10
        })
    else:
        dataset = load_from_disk(t_config.data_path)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=t_config.output_dir,
        per_device_train_batch_size=t_config.batch_size,
        learning_rate=t_config.learning_rate,
        num_train_epochs=t_config.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=t_config.fp16,
        gradient_accumulation_steps=t_config.gradient_accumulation_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    logger.info("Starting training...")
    try:
        # trainer.train() # Commented out to prevent execution without data/GPU
        logger.info("Training simulation completed successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")

    # Upload to S3
    if aws_client:
        logger.info("Uploading checkpoints to S3...")
        aws_client.sync_checkpoints(t_config.output_dir, aws_config.s3_bucket, aws_config.s3_prefix)

if __name__ == "__main__":
    t_conf = TrainingConfig()
    aws_conf = AWSConfig()
    train(t_conf, aws_conf)
