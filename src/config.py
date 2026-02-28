from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    data_path: str = "data/processed_dataset"
    output_dir: str = "checkpoints"
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05

@dataclass
class AWSConfig:
    s3_bucket: str = "my-llm-checkpoints"
    s3_prefix: str = "experiments/sft-001"
    region_name: str = "us-west-2"
    use_s3_checkpointing: bool = True
