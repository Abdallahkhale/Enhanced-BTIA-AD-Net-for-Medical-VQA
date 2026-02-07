"""
Configuration for Enhanced BTIA-AD Net
Medical Visual Question Answering
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Vision encoder
    vision_model: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    vision_dim: int = 512  # BioMedCLIP uses 512D embeddings
    image_size: int = 224
    
    # Text encoder
    text_model: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    text_dim: int = 512  # BioMedCLIP uses 512D embeddings
    max_question_len: int = 77
    
    # Answer distillation
    top_k: int = 10
    use_semantic_distillation: bool = True
    
    # Fusion
    fusion_dim: int = 512  # Match encoder output
    fusion_heads: int = 8
    fusion_layers: int = 2
    dropout: float = 0.1
    
    # Classification (separate heads for closed/open)
    use_dual_heads: bool = True



@dataclass
class TrainingConfig:
    """Training configuration for reaching paper accuracy"""
    # Data
    data_dir: str = "data/vqa_rad"
    batch_size: int = 16  # Standard batch size
    num_workers: int = 0  # 0 for Windows compatibility/debugging
    
    # Optimizer
    optimizer: str = "AdamW"
    learning_rate: float = 2e-5  # Standard CLIP fine-tune LR
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1e-7
    
    # Optimized Hyperparameters for BioMedCLIP Fine-tuning (VQA-RAD + PathVQA)
    batch_size: int = 32        # Increased from default (RTX 5070 Ti should handle 32)
    epochs: int = 25            # Reduced from 50 (since dataset is 10x larger)
    learning_rate: float = 5e-4 # Higher initial LR for heads
    weight_decay: float = 0.05  # Stronger regularization
    dropout: float = 0.3        # Prevent overfitting
    
    # Gradient Accumulation (Effective Batch Size = 32 * 1 = 32)
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Data Split
    use_full_train_data: bool = True  # Train on 100% of train.json
    use_pathvqa: bool = True          # Augment with PathVQA (20k samples)
    
    # Freezing strategy (3-Stage)
    # Stage 1: Heads only (Epochs 1-5)
    # Stage 2: Unfreeze Text (Epochs 6-9)
    # Stage 3: Unfreeze Vision (Epochs 10+)
    freeze_vision_epochs: int = 9
    freeze_text_epochs: int = 5
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 5
    
    # Logging
    log_dir: str = "logs"
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "btia-ad-net"
    
    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset configuration"""
    # VQA-RAD specifics
    train_file: str = "train.json"
    test_file: str = "test.json"
    images_dir: str = "images"
    
    # Preprocessing
    image_size: int = 224
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Augmentation
    use_augmentation: bool = True
    augment_prob: float = 0.5


@dataclass
class Config:
    """Main configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Device
    device: str = "cuda"
    
    # Paths
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    
    def __post_init__(self):
        # Create directories
        os.makedirs(self.training.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.project_root, self.training.data_dir), exist_ok=True)


def get_config() -> Config:
    """Get default configuration"""
    return Config()


if __name__ == "__main__":
    config = get_config()
    print(f"Model: {config.model.vision_model}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
