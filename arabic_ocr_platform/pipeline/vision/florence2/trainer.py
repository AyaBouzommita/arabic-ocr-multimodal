"""LoRA fine-tuning loop for Florence-2 object detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.dataset import FlorenceODDataset, ensure_florence_annotations


class Florence2Trainer:
    """Fine-tune Florence-2 on the shared YOLO dataset."""

    TARGET_MODULES = [
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "linear",
        "Conv2d",
        "lm_head",
        "fc2",
    ]

    def __init__(self, config: Optional[Florence2Config] = None):
        self.config = config or Florence2Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_annotations(
        self,
        limit_per_split: Optional[int] = None,
    ) -> Dict[str, Path]:
        return ensure_florence_annotations(
            self.config.annotations_dir,
            splits=["train", "valid"],
            limit_per_split=limit_per_split,
            dataset_yaml=self.config.dataset_yaml,
        )

    def _build_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.checkpoint,
            trust_remote_code=True,
            revision=self.config.revision,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(
            self.config.checkpoint,
            trust_remote_code=True,
            revision=self.config.revision,
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.TARGET_MODULES,
            task_type="CAUSAL_LM",
            lora_dropout=self.config.lora_dropout,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        return model, processor

    def _collate(self, processor, batch):
        images = [item["image"] for item in batch]
        prefixes = [item["prefix"] for item in batch]
        suffixes = [item["suffix"] for item in batch]

        inputs = processor(text=prefixes, images=images, return_tensors="pt", padding=True)
        labels = processor.tokenizer(
            text=suffixes,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).input_ids.to(self.device)
        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "pixel_values": inputs["pixel_values"].to(self.device),
            "labels": labels,
        }

    def train(
        self,
        limit_per_split: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> Dict:
        annotation_paths = self.prepare_annotations(limit_per_split=limit_per_split)
        train_dataset = FlorenceODDataset(annotation_paths["train"])
        val_dataset = FlorenceODDataset(annotation_paths["valid"])

        model, processor = self._build_model()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: self._collate(processor, batch),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: self._collate(processor, batch),
        )

        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        num_epochs = epochs or self.config.epochs
        num_training_steps = num_epochs * max(len(train_loader), 1)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        history = {"train_loss": [], "val_loss": []}
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            avg_train = train_loss / max(len(train_loader), 1)
            history["train_loss"].append(round(avg_train, 4))

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    val_loss += outputs.loss.item()
            avg_val = val_loss / max(len(val_loader), 1)
            history["val_loss"].append(round(avg_val, 4))

        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.config.model_dir)
        processor.save_pretrained(self.config.model_dir)

        summary = {
            "checkpoint": self.config.checkpoint,
            "epochs": num_epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "device": str(self.device),
            "history": history,
            "model_dir": str(self.config.model_dir),
        }
        (self.config.model_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
