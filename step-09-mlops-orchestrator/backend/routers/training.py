"""DeepSpeed distributed training endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from ...ai_engine.deepspeed_launcher import (
    DeepSpeedLauncher, DeepSpeedConfig, LaunchResult,
    ZeROStage, OffloadDevice,
)

router = APIRouter()
launcher = DeepSpeedLauncher()


class TrainingLaunchRequest(BaseModel):
    """Request to launch a DeepSpeed training job."""
    training_script: str = Field(..., description="Path to training script")
    model_name: str = Field(default="custom_model")
    script_args: Optional[List[str]] = None

    # ZeRO config
    zero_stage: int = Field(default=3, ge=0, le=3)
    offload_optimizer: str = Field(default="cpu")
    offload_param: str = Field(default="cpu")

    # Batch / precision
    micro_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    fp16: bool = True
    bf16: bool = False

    # LR
    learning_rate: float = Field(default=3e-5, gt=0)
    warmup_steps: int = 100
    total_steps: int = 10000

    # Hardware
    num_gpus: int = Field(default=1, ge=1)
    num_nodes: int = Field(default=1, ge=1)

    # Options
    activation_checkpointing: bool = True
    dry_run: bool = Field(default=True, description="If true, only generates config without launching")


class PresetLaunchRequest(BaseModel):
    """Launch from a preset configuration (7b, 13b, 70b)."""
    preset: str = Field(..., description="Preset name: 7b | 13b | 70b")
    training_script: str = Field(...)
    script_args: Optional[List[str]] = None
    dry_run: bool = True


@router.post("/launch", response_model=LaunchResult)
async def launch_training(request: TrainingLaunchRequest):
    """Configure and launch a DeepSpeed ZeRO-3 training job."""
    config = DeepSpeedConfig(
        model_name=request.model_name,
        zero_stage=ZeROStage(request.zero_stage),
        offload_optimizer=OffloadDevice(request.offload_optimizer),
        offload_param=OffloadDevice(request.offload_param),
        train_micro_batch_size_per_gpu=request.micro_batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        fp16_enabled=request.fp16,
        bf16_enabled=request.bf16,
        learning_rate=request.learning_rate,
        warmup_steps=request.warmup_steps,
        total_training_steps=request.total_steps,
        num_gpus=request.num_gpus,
        num_nodes=request.num_nodes,
        activation_checkpointing=request.activation_checkpointing,
    )

    result = launcher.launch(
        config, request.training_script,
        request.script_args, request.dry_run
    )
    return result


@router.post("/launch/preset", response_model=LaunchResult)
async def launch_from_preset(request: PresetLaunchRequest):
    """Launch a training job using a pre-configured model-size preset."""
    presets = launcher.presets()
    if request.preset not in presets:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{request.preset}'. Available: {list(presets.keys())}"
        )

    config = presets[request.preset]
    result = launcher.launch(
        config, request.training_script,
        request.script_args, request.dry_run
    )
    return result


@router.get("/presets")
async def list_presets():
    """List available DeepSpeed configuration presets."""
    presets = launcher.presets()
    return {
        preset_name: {
            "zero_stage": cfg.zero_stage.value,
            "offload_optimizer": cfg.offload_optimizer.value,
            "offload_param": cfg.offload_param.value,
            "micro_batch_size": cfg.train_micro_batch_size_per_gpu,
            "gradient_accumulation": cfg.gradient_accumulation_steps,
            "precision": "bf16" if cfg.bf16_enabled else "fp16",
            "num_gpus": cfg.num_gpus,
            "num_nodes": cfg.num_nodes,
            "activation_checkpointing": cfg.activation_checkpointing,
        }
        for preset_name, cfg in presets.items()
    }


@router.post("/config/generate")
async def generate_config(request: TrainingLaunchRequest):
    """Generate a DeepSpeed JSON config without launching."""
    config = DeepSpeedConfig(
        model_name=request.model_name,
        zero_stage=ZeROStage(request.zero_stage),
        offload_optimizer=OffloadDevice(request.offload_optimizer),
        offload_param=OffloadDevice(request.offload_param),
        train_micro_batch_size_per_gpu=request.micro_batch_size,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        fp16_enabled=request.fp16,
        bf16_enabled=request.bf16,
        learning_rate=request.learning_rate,
        warmup_steps=request.warmup_steps,
        total_training_steps=request.total_steps,
        num_gpus=request.num_gpus,
        num_nodes=request.num_nodes,
        activation_checkpointing=request.activation_checkpointing,
    )

    ds_config = launcher.generate_config(config)
    command = launcher.build_launch_command(config, request.training_script, request.script_args)

    effective_batch = (
        config.train_micro_batch_size_per_gpu *
        config.gradient_accumulation_steps *
        config.num_gpus * config.num_nodes
    )

    return {
        "deepspeed_config": ds_config,
        "launch_command": command,
        "effective_batch_size": effective_batch,
    }
