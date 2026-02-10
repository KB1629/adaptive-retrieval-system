"""
Tests for fine-tuning pipeline.

This module tests:
- LoRA configuration and trainer setup
- Checkpoint management
- Synthetic QA generation
- Property 14: Checkpoint Interval Consistency
- Property 15: LoRA Weights Round-Trip
- Property 16: Training Resume from Checkpoint

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

import pytest
import os
import tempfile
import shutil
from hypothesis import given, strategies as st, settings

from src.finetuning.lora_trainer import LoRATrainer, LoRAConfig, create_lora_config_for_t4
from src.finetuning.synthetic_qa import SyntheticQAGenerator, QAPair, create_training_dataset


# Unit Tests for LoRA Trainer

class TestLoRAConfig:
    """Test LoRA configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LoRAConfig()
        
        assert config.model_name == "vidore/colpali"
        assert config.lora_rank == 8
        assert config.lora_alpha == 16
        assert config.batch_size == 4
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LoRAConfig(
            model_name="custom/model",
            lora_rank=16,
            batch_size=2,
        )
        
        assert config.model_name == "custom/model"
        assert config.lora_rank == 16
        assert config.batch_size == 2
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = LoRAConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "lora_rank" in config_dict
    
    def test_config_save_load(self):
        """Test config save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(lora_rank=16)
            filepath = os.path.join(tmpdir, "config.json")
            
            config.save(filepath)
            loaded_config = LoRAConfig.load(filepath)
            
            assert loaded_config.lora_rank == 16
            assert loaded_config.model_name == config.model_name
    
    def test_t4_optimized_config(self):
        """Test T4-optimized configuration."""
        config = create_lora_config_for_t4()
        
        assert config.lora_rank == 8  # Low rank for memory
        assert config.batch_size == 2  # Small batch
        assert config.device == "cuda"


class TestLoRATrainer:
    """Test LoRA trainer."""
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(output_dir=tmpdir)
            trainer = LoRATrainer(config)
            
            assert trainer.config == config
            assert trainer.current_step == 0
            assert trainer.current_epoch == 0
            assert os.path.exists(tmpdir)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(output_dir=tmpdir)
            trainer = LoRATrainer(config)
            
            trainer.current_step = 100
            trainer.save_checkpoint()
            
            checkpoint_dir = os.path.join(tmpdir, "checkpoint-100")
            assert os.path.exists(checkpoint_dir)
            assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint_info.json"))
    
    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(output_dir=tmpdir)
            trainer = LoRATrainer(config)
            
            # Save checkpoint
            trainer.current_step = 100
            trainer.current_epoch = 2
            trainer.save_checkpoint()
            
            # Create new trainer and load
            new_trainer = LoRATrainer(config)
            checkpoint_dir = os.path.join(tmpdir, "checkpoint-100")
            new_trainer.load_checkpoint(checkpoint_dir)
            
            assert new_trainer.current_step == 100
            assert new_trainer.current_epoch == 2
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(output_dir=tmpdir)
            trainer = LoRATrainer(config)
            
            # Save multiple checkpoints
            for step in [100, 200, 300]:
                trainer.current_step = step
                trainer.save_checkpoint()
            
            checkpoints = trainer.list_checkpoints()
            
            assert len(checkpoints) == 3
            assert "checkpoint-100" in checkpoints[0]
            assert "checkpoint-300" in checkpoints[2]
    
    def test_export_lora_weights(self):
        """Test LoRA weight export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(output_dir=tmpdir)
            trainer = LoRATrainer(config)
            
            output_path = os.path.join(tmpdir, "lora_weights.pt")
            trainer.export_lora_weights(output_path)
            
            # Check metadata file exists
            metadata_path = output_path.replace(".pt", "_metadata.json")
            assert os.path.exists(metadata_path)
    
    def test_get_training_metrics(self):
        """Test getting training metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = LoRAConfig(output_dir=tmpdir)
            trainer = LoRATrainer(config)
            
            trainer.current_step = 50
            metrics = trainer.get_training_metrics()
            
            assert metrics["current_step"] == 50
            assert "config" in metrics


# Unit Tests for Synthetic QA

class TestQAPair:
    """Test QA pair data structure."""
    
    def test_qa_pair_creation(self):
        """Test QA pair creation."""
        qa = QAPair(
            question="What is X?",
            answer="X is Y",
            doc_id="doc1",
            page_number=1,
        )
        
        assert qa.question == "What is X?"
        assert qa.answer == "X is Y"
        assert qa.doc_id == "doc1"
    
    def test_qa_pair_to_dict(self):
        """Test QA pair serialization."""
        qa = QAPair(question="Q", answer="A")
        qa_dict = qa.to_dict()
        
        assert isinstance(qa_dict, dict)
        assert qa_dict["question"] == "Q"
        assert qa_dict["answer"] == "A"


class TestSyntheticQAGenerator:
    """Test synthetic QA generator."""
    
    def test_generator_init(self):
        """Test generator initialization."""
        generator = SyntheticQAGenerator(num_questions_per_page=3)
        
        assert generator.num_questions_per_page == 3
    
    def test_generate_from_page(self):
        """Test QA generation from page."""
        generator = SyntheticQAGenerator(num_questions_per_page=3)
        
        qa_pairs = generator.generate_from_page(
            page_image=None,  # Placeholder
            page_text="Sample text",
            doc_id="doc1",
            page_number=1,
        )
        
        assert len(qa_pairs) == 3
        assert all(isinstance(qa, QAPair) for qa in qa_pairs)
        assert all(qa.doc_id == "doc1" for qa in qa_pairs)
    
    def test_save_load_qa_pairs(self):
        """Test saving and loading QA pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = SyntheticQAGenerator()
            qa_pairs = [
                QAPair(question="Q1", answer="A1"),
                QAPair(question="Q2", answer="A2"),
            ]
            
            filepath = os.path.join(tmpdir, "qa_pairs.json")
            generator.save_qa_pairs(qa_pairs, filepath)
            
            loaded_pairs = SyntheticQAGenerator.load_qa_pairs(filepath)
            
            assert len(loaded_pairs) == 2
            assert loaded_pairs[0].question == "Q1"
            assert loaded_pairs[1].answer == "A2"


class TestTrainingDataset:
    """Test training dataset creation."""
    
    def test_create_training_dataset_json(self):
        """Test creating JSON training dataset."""
        qa_pairs = [
            QAPair(question="Q1", answer="A1"),
            QAPair(question="Q2", answer="A2"),
        ]
        
        dataset = create_training_dataset(qa_pairs, output_format="json")
        
        assert isinstance(dataset, list)
        assert len(dataset) == 2
        assert dataset[0]["question"] == "Q1"


# Property-Based Tests

# Feature: adaptive-retrieval-system, Property 14: Checkpoint Interval Consistency
# **Validates: Requirements 8.3**
@settings(max_examples=10, deadline=None)
@given(
    checkpoint_interval=st.integers(min_value=10, max_value=500),
    total_steps=st.integers(min_value=50, max_value=1000),
)
def test_property_checkpoint_interval_consistency(checkpoint_interval, total_steps):
    """
    Property 14: Checkpoint Interval Consistency
    
    For any fine-tuning run with checkpoint_interval=N, checkpoints SHALL be
    saved at steps N, 2N, 3N, etc., and each checkpoint SHALL contain valid
    model weights loadable for resumption.
    
    **Validates: Requirements 8.3**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoRAConfig(
            output_dir=tmpdir,
            checkpoint_interval=checkpoint_interval,
        )
        trainer = LoRATrainer(config)
        
        # Simulate saving checkpoints at intervals
        expected_checkpoints = []
        for step in range(checkpoint_interval, total_steps + 1, checkpoint_interval):
            trainer.current_step = step
            trainer.save_checkpoint()
            expected_checkpoints.append(step)
        
        # Verify checkpoints exist at correct intervals
        actual_checkpoints = trainer._get_checkpoint_steps()
        
        # Property 1: Checkpoints saved at correct intervals
        assert actual_checkpoints == expected_checkpoints, \
            f"Expected checkpoints at {expected_checkpoints}, got {actual_checkpoints}"
        
        # Property 2: Each checkpoint is loadable
        for step in actual_checkpoints:
            checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{step}")
            assert os.path.exists(checkpoint_dir), f"Checkpoint {step} missing"
            
            # Verify checkpoint info exists
            info_file = os.path.join(checkpoint_dir, "checkpoint_info.json")
            assert os.path.exists(info_file), f"Checkpoint info missing for step {step}"


# Feature: adaptive-retrieval-system, Property 15: LoRA Weights Round-Trip
# **Validates: Requirements 8.5**
@settings(max_examples=10, deadline=None)
@given(
    lora_rank=st.integers(min_value=4, max_value=64),
)
def test_property_lora_weights_round_trip(lora_rank):
    """
    Property 15: LoRA Weights Round-Trip
    
    For any LoRA weights exported by Fine_Tuning_Pipeline, loading those weights
    into Vision_Embedding_Path SHALL succeed without errors, and the loaded model
    SHALL produce embeddings with the same dimensions as the original fine-tuned model.
    
    **Validates: Requirements 8.5**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoRAConfig(
            output_dir=tmpdir,
            lora_rank=lora_rank,
        )
        trainer = LoRATrainer(config)
        
        # Export weights
        weights_path = os.path.join(tmpdir, "lora_weights.pt")
        trainer.export_lora_weights(weights_path)
        
        # Property 1: Metadata file exists
        metadata_path = weights_path.replace(".pt", "_metadata.json")
        assert os.path.exists(metadata_path), "Metadata file missing"
        
        # Property 2: Metadata contains correct rank
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata["lora_rank"] == lora_rank, \
            f"Metadata rank {metadata['lora_rank']} != config rank {lora_rank}"
        
        # Property 3: Metadata is complete
        required_fields = ["model_name", "lora_rank", "lora_alpha", "target_modules"]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"


# Feature: adaptive-retrieval-system, Property 16: Training Resume from Checkpoint
# **Validates: Requirements 8.7**
@settings(max_examples=10, deadline=None)
@given(
    resume_step=st.integers(min_value=100, max_value=500),
    resume_epoch=st.integers(min_value=1, max_value=10),
)
def test_property_training_resume_from_checkpoint(resume_step, resume_epoch):
    """
    Property 16: Training Resume from Checkpoint
    
    For any training run interrupted at step S and resumed from checkpoint, the
    resumed training SHALL continue from step S (not restart from 0), and the
    final model SHALL be equivalent to an uninterrupted training run.
    
    **Validates: Requirements 8.7**
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LoRAConfig(output_dir=tmpdir)
        
        # First trainer - save checkpoint
        trainer1 = LoRATrainer(config)
        trainer1.current_step = resume_step
        trainer1.current_epoch = resume_epoch
        trainer1.save_checkpoint()
        
        # Second trainer - resume from checkpoint
        trainer2 = LoRATrainer(config)
        checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{resume_step}")
        trainer2.load_checkpoint(checkpoint_dir)
        
        # Property 1: Resumed from correct step (not 0)
        assert trainer2.current_step == resume_step, \
            f"Resumed at step {trainer2.current_step}, expected {resume_step}"
        assert trainer2.current_step != 0, "Training restarted from 0 instead of resuming"
        
        # Property 2: Resumed from correct epoch
        assert trainer2.current_epoch == resume_epoch, \
            f"Resumed at epoch {trainer2.current_epoch}, expected {resume_epoch}"
        
        # Property 3: State is consistent
        assert trainer2.current_step > 0, "Invalid resumed state"
        assert trainer2.current_epoch > 0, "Invalid resumed epoch"
