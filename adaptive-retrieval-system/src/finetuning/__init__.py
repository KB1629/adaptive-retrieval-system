"""
Fine-tuning pipeline for vision models.

This module provides LoRA fine-tuning capabilities for domain adaptation:
- LoRA trainer for SigLIP/ColPali models
- Checkpoint management
- Synthetic QA generation
- Weight export

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

from .lora_trainer import LoRATrainer, LoRAConfig

__all__ = ["LoRATrainer", "LoRAConfig"]
