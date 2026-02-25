"""
DeepSpeed ZeRO-3 Distributed Training Launcher.

Configures and launches DeepSpeed ZeRO Stage 3 training with:
- Full parameter + gradient + optimizer state partitioning
- CPU offloading for optimizer and parameters
- Automatic micro-batch sizing
- Configurable pipeline/tensor parallelism
- Dynamic config generation for different model sizes
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum


class ZeROStage(int, Enum):
    DISABLED = 0
    OPTIMIZER_STATE = 1
    GRADIENT_PARTITIONING = 2
    FULL_PARTITIONING = 3


class OffloadDevice(str, Enum):
    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"


class DeepSpeedConfig(BaseModel):
    """Configuration for DeepSpeed ZeRO-3 training."""
    # ZeRO configuration
    zero_stage: ZeROStage = ZeROStage.FULL_PARTITIONING
    offload_optimizer: OffloadDevice = OffloadDevice.CPU
    offload_param: OffloadDevice = OffloadDevice.CPU

    # Batch sizing
    train_micro_batch_size_per_gpu: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    gradient_clipping: float = Field(default=1.0, ge=0)

    # Precision
    fp16_enabled: bool = True
    bf16_enabled: bool = False
    fp16_loss_scale: float = 0
    fp16_loss_scale_window: int = 1000

    # Learning rate
    learning_rate: float = Field(default=3e-5, gt=0)
    lr_scheduler_type: str = "WarmupDecayLR"
    warmup_steps: int = 100
    total_training_steps: int = 10000

    # Communication
    communication_data_type: str = "fp16"
    prescale_gradients: bool = False
    gradient_predivide_factor: float = 1.0

    # Activation checkpointing
    activation_checkpointing: bool = True
    contiguous_memory_optimization: bool = True
    cpu_checkpointing: bool = False

    # ZeRO-3 specific
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000
    stage3_prefetch_bucket_size: int = 500_000_000
    stage3_param_persistence_threshold: int = 100_000
    stage3_gather_16bit_weights_on_model_save: bool = True
    reduce_bucket_size: int = 500_000_000

    # Advanced
    elastic_training: bool = False
    wall_clock_breakdown: bool = True
    dump_state: bool = False

    # Model info
    model_name: str = "custom_model"
    num_gpus: int = Field(default=1, ge=1)
    num_nodes: int = Field(default=1, ge=1)
    master_addr: str = "localhost"
    master_port: int = 29500


class LaunchResult(BaseModel):
    """Result of a DeepSpeed launch."""
    job_id: str
    config_path: str
    command: str
    status: str = "launched"
    num_gpus: int
    num_nodes: int
    effective_batch_size: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Dict[str, Any] = Field(default_factory=dict)


class DeepSpeedLauncher:
    """
    Configures and launches DeepSpeed ZeRO-3 distributed training.

    Generates DeepSpeed JSON configs from structured parameters,
    builds the launch command, and manages the training process.
    """

    def __init__(self, deepspeed_path: str = "deepspeed"):
        self.deepspeed_path = deepspeed_path

    def generate_config(self, config: DeepSpeedConfig) -> Dict[str, Any]:
        """
        Generate a complete DeepSpeed JSON configuration.

        Args:
            config: DeepSpeedConfig with training parameters

        Returns:
            Dict suitable for writing as deepspeed JSON config
        """
        ds_config = {
            "train_micro_batch_size_per_gpu": config.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "gradient_clipping": config.gradient_clipping,
            "steps_per_print": 100,
            "wall_clock_breakdown": config.wall_clock_breakdown,
            "dump_state": config.dump_state,

            # ZeRO configuration
            "zero_optimization": {
                "stage": config.zero_stage.value,
                "allgather_partitions": True,
                "allgather_bucket_size": config.reduce_bucket_size,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": config.reduce_bucket_size,
                "contiguous_gradients": True,
                "round_robin_gradients": True,
            },

            # Learning rate scheduler
            "scheduler": {
                "type": config.lr_scheduler_type,
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": config.learning_rate,
                    "warmup_num_steps": config.warmup_steps,
                    "total_num_steps": config.total_training_steps,
                },
            },

            # Optimizer
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            },

            # Communication
            "communication_data_type": config.communication_data_type,
            "prescale_gradients": config.prescale_gradients,
            "gradient_predivide_factor": config.gradient_predivide_factor,
        }

        # Precision settings
        if config.bf16_enabled:
            ds_config["bf16"] = {"enabled": True}
        elif config.fp16_enabled:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": config.fp16_loss_scale,
                "loss_scale_window": config.fp16_loss_scale_window,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }

        # ZeRO-3 specific settings
        if config.zero_stage == ZeROStage.FULL_PARTITIONING:
            ds_config["zero_optimization"].update({
                "stage3_max_live_parameters": config.stage3_max_live_parameters,
                "stage3_max_reuse_distance": config.stage3_max_reuse_distance,
                "stage3_prefetch_bucket_size": config.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": config.stage3_param_persistence_threshold,
                "stage3_gather_16bit_weights_on_model_save": config.stage3_gather_16bit_weights_on_model_save,
                "sub_group_size": 1_000_000_000,
            })

        # Offloading
        if config.offload_optimizer != OffloadDevice.NONE:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": config.offload_optimizer.value,
                "pin_memory": True,
                "buffer_count": 4,
                "fast_init": False,
            }

        if config.offload_param != OffloadDevice.NONE:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": config.offload_param.value,
                "pin_memory": True,
                "buffer_count": 5,
                "buffer_size": 1_000_000_000,
                "max_in_cpu": 1_000_000_000,
            }

        # Activation checkpointing
        if config.activation_checkpointing:
            ds_config["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": config.cpu_checkpointing,
                "contiguous_memory_optimization": config.contiguous_memory_optimization,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            }

        # Elastic training
        if config.elastic_training:
            ds_config["elasticity"] = {
                "enabled": True,
                "max_train_batch_size": (
                    config.train_micro_batch_size_per_gpu *
                    config.gradient_accumulation_steps *
                    config.num_gpus * 2
                ),
                "min_train_batch_size": config.train_micro_batch_size_per_gpu,
                "min_gpus": 1,
                "max_gpus": config.num_gpus * config.num_nodes,
                "prefer_larger_batch_size": True,
            }

        return ds_config

    def write_config(self, config: DeepSpeedConfig,
                     output_path: Optional[str] = None) -> str:
        """Write DeepSpeed config to JSON file."""
        ds_config = self.generate_config(config)

        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"ds_config_{config.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )

        with open(output_path, "w") as f:
            json.dump(ds_config, f, indent=2)

        return output_path

    def build_launch_command(self, config: DeepSpeedConfig,
                             training_script: str,
                             script_args: Optional[List[str]] = None,
                             config_path: Optional[str] = None) -> str:
        """
        Build the DeepSpeed launch command string.

        Args:
            config: DeepSpeedConfig
            training_script: path to training script
            script_args: additional script arguments
            config_path: path to DS config JSON

        Returns:
            Full command string
        """
        if config_path is None:
            config_path = self.write_config(config)

        parts = [self.deepspeed_path]

        # Multi-node
        if config.num_nodes > 1:
            parts.extend([
                f"--num_nodes={config.num_nodes}",
                f"--master_addr={config.master_addr}",
                f"--master_port={config.master_port}",
            ])

        # GPUs
        parts.append(f"--num_gpus={config.num_gpus}")

        # Training script
        parts.append(training_script)

        # DeepSpeed config
        parts.append(f"--deepspeed_config={config_path}")

        # Additional args
        if script_args:
            parts.extend(script_args)

        return " ".join(parts)

    def launch(self, config: DeepSpeedConfig,
               training_script: str,
               script_args: Optional[List[str]] = None,
               dry_run: bool = False) -> LaunchResult:
        """
        Launch a DeepSpeed ZeRO-3 training job.

        Args:
            config: training configuration
            training_script: path to training script
            script_args: additional arguments for training script
            dry_run: if True, generate command but don't execute

        Returns:
            LaunchResult with job details
        """
        config_path = self.write_config(config)
        command = self.build_launch_command(
            config, training_script, script_args, config_path
        )

        effective_batch = (
            config.train_micro_batch_size_per_gpu *
            config.gradient_accumulation_steps *
            config.num_gpus *
            config.num_nodes
        )

        job_id = f"ds_{config.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        result = LaunchResult(
            job_id=job_id,
            config_path=config_path,
            command=command,
            num_gpus=config.num_gpus * config.num_nodes,
            num_nodes=config.num_nodes,
            effective_batch_size=effective_batch,
            details={
                "zero_stage": config.zero_stage.value,
                "offload_optimizer": config.offload_optimizer.value,
                "offload_param": config.offload_param.value,
                "precision": "bf16" if config.bf16_enabled else "fp16" if config.fp16_enabled else "fp32",
                "activation_checkpointing": config.activation_checkpointing,
                "dry_run": dry_run,
            },
        )

        if dry_run:
            result.status = "dry_run"
            return result

        try:
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "MASTER_ADDR": config.master_addr,
                     "MASTER_PORT": str(config.master_port)},
            )
            result.status = "launched"
            result.details["pid"] = process.pid
        except Exception as e:
            result.status = "failed"
            result.details["error"] = str(e)

        return result

    @staticmethod
    def presets() -> Dict[str, DeepSpeedConfig]:
        """Pre-configured settings for common model sizes."""
        return {
            "7b": DeepSpeedConfig(
                model_name="7b_model",
                train_micro_batch_size_per_gpu=2,
                gradient_accumulation_steps=16,
                zero_stage=ZeROStage.FULL_PARTITIONING,
                offload_optimizer=OffloadDevice.CPU,
                offload_param=OffloadDevice.NONE,
                fp16_enabled=True,
                num_gpus=4,
            ),
            "13b": DeepSpeedConfig(
                model_name="13b_model",
                train_micro_batch_size_per_gpu=1,
                gradient_accumulation_steps=32,
                zero_stage=ZeROStage.FULL_PARTITIONING,
                offload_optimizer=OffloadDevice.CPU,
                offload_param=OffloadDevice.CPU,
                fp16_enabled=True,
                num_gpus=8,
            ),
            "70b": DeepSpeedConfig(
                model_name="70b_model",
                train_micro_batch_size_per_gpu=1,
                gradient_accumulation_steps=64,
                zero_stage=ZeROStage.FULL_PARTITIONING,
                offload_optimizer=OffloadDevice.CPU,
                offload_param=OffloadDevice.CPU,
                bf16_enabled=True,
                fp16_enabled=False,
                activation_checkpointing=True,
                cpu_checkpointing=True,
                num_gpus=8,
                num_nodes=2,
            ),
        }
