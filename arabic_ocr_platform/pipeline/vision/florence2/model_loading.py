"""Shared Florence-2 model loading helpers (low-VRAM aware)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config

LOW_VRAM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FULL_TARGET_MODULES = [
    "q_proj",
    "o_proj",
    "k_proj",
    "v_proj",
    "linear",
    "Conv2d",
    "lm_head",
    "fc2",
]


def gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def should_use_low_vram(config: Optional[Florence2Config] = None) -> bool:
    cfg = config or Florence2Config()
    if cfg.force_low_vram:
        return True
    if cfg.force_full_precision:
        return False
    return gpu_vram_gb() <= cfg.low_vram_threshold_gb


def freeze_vision_encoder(model) -> None:
    """Freeze vision backbone to reduce VRAM during LoRA training."""
    for attr in ("vision_tower", "image_projection", "visual"):
        module = getattr(model, attr, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False

    base = getattr(model, "model", model)
    for attr in ("vision_tower", "image_projection", "visual", "encoder"):
        module = getattr(base, attr, None)
        if module is not None and attr != "encoder":
            for param in module.parameters():
                param.requires_grad = False


def load_florence2_model(
    checkpoint: str,
    revision: str,
    config: Optional[Florence2Config] = None,
    for_training: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
):
    """Load Florence-2 with optional 4-bit QLoRA for GPUs <= 6 GB VRAM."""
    cfg = config or Florence2Config()
    low_vram = should_use_low_vram(cfg)
    if low_vram:
        cfg.apply_low_vram_defaults()

    load_kwargs = {
        "trust_remote_code": True,
        "revision": revision,
        "attn_implementation": "eager",
    }

    if low_vram:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {0: "3500MiB", "cpu": "12GiB"}
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(checkpoint, **load_kwargs)

    if for_training:
        freeze_vision_encoder(model)

    if for_training and low_vram:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    elif for_training and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if for_training:
        target_modules = LOW_VRAM_TARGET_MODULES if low_vram else FULL_TARGET_MODULES
        lora_config = LoraConfig(
            r=lora_r if not low_vram else cfg.lora_r,
            lora_alpha=lora_alpha if not low_vram else cfg.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=lora_dropout,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)

    if not low_vram:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    return model, low_vram


def load_florence2_for_inference(
    checkpoint: str,
    revision: str,
    config: Optional[Florence2Config] = None,
    adapter_dir: Optional[str] = None,
):
    """Load Florence-2 for inference, including fine-tuned LoRA adapters."""
    from pathlib import Path

    from peft import AutoPeftModelForCausalLM

    cfg = config or Florence2Config()
    low_vram = should_use_low_vram(cfg)
    adapter_path = Path(adapter_dir) if adapter_dir else None

    if adapter_path and (adapter_path / "adapter_config.json").exists():
        load_kwargs = {
            "trust_remote_code": True,
            "revision": revision,
            "attn_implementation": "eager",
        }
        if low_vram:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
        model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), **load_kwargs)
        if not low_vram:
            model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model, low_vram

    return load_florence2_model(
        checkpoint=checkpoint,
        revision=revision,
        config=cfg,
        for_training=False,
    )


def load_florence2_processor(checkpoint: str, revision: str) -> AutoProcessor:
    return AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        revision=revision,
    )
