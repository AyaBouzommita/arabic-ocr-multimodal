"""LoRA fine-tuning loop for Florence-2 object detection."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from arabic_ocr_platform.pipeline.vision.florence2.config import Florence2Config
from arabic_ocr_platform.pipeline.vision.florence2.dataset import FlorenceODDataset, ensure_florence_annotations
from arabic_ocr_platform.pipeline.vision.florence2.model_loading import (
    load_florence2_model,
    load_florence2_processor,
    should_use_low_vram,
)


class Florence2Trainer:
    """Fine-tune Florence-2 on the shared YOLO dataset."""

    def __init__(self, config: Optional[Florence2Config] = None):
        self.config = config or Florence2Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.low_vram = should_use_low_vram(self.config)

    def prepare_annotations(
        self,
        limit_per_split: Optional[int] = None,
    ) -> Dict[str, Path]:
        return ensure_florence_annotations(
            self.config.annotations_dir,
            splits=["train", "valid"],
            limit_per_split=limit_per_split,
            dataset_yaml=self.config.dataset_yaml,
            max_boxes_per_image=self.config.max_boxes_per_image,
        )

    def _resize_image(self, image: Image.Image) -> Image.Image:
        max_size = self.config.max_image_size
        width, height = image.size
        scale = min(max_size / max(width, height), 1.0)
        if scale >= 1.0:
            return image
        return image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)

    def _build_model(self):
        model, self.low_vram = load_florence2_model(
            checkpoint=self.config.checkpoint,
            revision=self.config.revision,
            config=self.config,
            for_training=True,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )
        processor = load_florence2_processor(self.config.checkpoint, self.config.revision)
        if self.low_vram:
            print(
                f"[Low-VRAM Mode] 4-bit QLoRA | image {self.config.max_image_size}px | "
                f"seq {self.config.max_seq_length} | max {self.config.max_boxes_per_image} boxes/img"
            )
        return model, processor

    def _collate(self, processor, batch):
        images = [self._resize_image(item["image"]) for item in batch]
        prefixes = [item["prefix"] for item in batch]
        suffixes = [item["suffix"] for item in batch]

        inputs = processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = processor.tokenizer(
            text=suffixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_token_type_ids=False,
        ).input_ids

        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        device = self.device
        return {
            "input_ids": inputs["input_ids"].to(device),
            "pixel_values": inputs["pixel_values"].to(device),
            "labels": labels.to(device),
        }

    def _evaluate_metrics(self, model, processor, limit_per_split: Optional[int] = None) -> Dict[str, float]:
        """Compute precision, recall, map50, map50_95 on the validation set."""
        from PIL import Image
        from arabic_ocr_platform.pipeline.vision.yolo_dataset import load_split_samples, class_names_from_config, load_dataset_config
        from arabic_ocr_platform.pipeline.vision.florence2.metrics import Detection, compute_map, compute_precision_recall
        from arabic_ocr_platform.pipeline.vision.florence2.engine import Florence2Detector
        
        dataset_cfg = load_dataset_config(self.config.dataset_yaml)
        class_map = class_names_from_config(dataset_cfg)
        allowed_classes = set(class_map.values())
        
        val_samples = load_split_samples("valid", limit=limit_per_split, dataset_yaml=self.config.dataset_yaml)
        
        pred_by_image = {}
        gt_by_image = {}
        
        model.eval()
        with torch.no_grad():
            for sample in val_samples:
                image_id = sample.image_path.stem
                gt_by_image[image_id] = [
                    Detection(
                        label=box.class_name,
                        bbox=list(box.to_xyxy_pixels(sample.image_width, sample.image_height)),
                        confidence=1.0,
                    )
                    for box in sample.boxes
                ]
                
                with Image.open(sample.image_path) as img:
                    image = img.convert("RGB")
                
                image = self._resize_image(image)
                inputs = processor(text=self.config.task_prompt, images=image, return_tensors="pt")
                device = self.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.cuda.amp.autocast(enabled=self.low_vram):
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=self.config.max_new_tokens,
                        num_beams=self.config.num_beams,
                    )
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    generated_text,
                    task=self.config.task_prompt,
                    image_size=image.size,
                )
                
                objects = Florence2Detector._parse_objects(parsed, allowed_classes)
                pred_by_image[image_id] = [
                    Detection(label=obj["label"], bbox=obj["bbox"], confidence=obj["confidence"])
                    for obj in objects
                ]
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        map_metrics = compute_map(
            pred_by_image,
            gt_by_image,
            class_names=sorted(allowed_classes),
            iou_thresholds=[0.5],
        )
        pr_metrics = compute_precision_recall(pred_by_image, gt_by_image, iou_threshold=0.5)
        
        return {
            "precision": pr_metrics["precision"],
            "recall": pr_metrics["recall"],
            "map50": map_metrics["map50"],
            "map50_95": map_metrics["map50_95"],
        }

    def train(
        self,
        limit_per_split: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> Dict:
        import time as time_lib

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # Initialize results.csv
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.config.model_dir / "results.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
                "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
                "val/box_loss,val/cls_loss,val/dfl_loss\n"
            )

        history = {"train_loss": [], "val_loss": []}
        for epoch in range(num_epochs):
            epoch_start = time_lib.perf_counter()
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                with torch.cuda.amp.autocast(enabled=self.low_vram):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                train_loss += loss.item()
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            avg_val = val_loss / max(len(val_loader), 1)
            history["val_loss"].append(round(avg_val, 4))
            
            epoch_time = time_lib.perf_counter() - epoch_start
            
            # Evaluate validation set metrics
            print(f"Epoch {epoch+1}/{num_epochs}: Running validation set evaluation...")
            val_metrics = self._evaluate_metrics(model, processor, limit_per_split=limit_per_split)
            
            # Write to results.csv
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch+1},{epoch_time:.4f},{avg_train:.4f},0.0,0.0,"
                    f"{val_metrics['precision']:.4f},{val_metrics['recall']:.4f},"
                    f"{val_metrics['map50']:.4f},{val_metrics['map50_95']:.4f},"
                    f"{avg_val:.4f},0.0,0.0\n"
                )
            
            gc.collect()

        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.config.model_dir)
        processor.save_pretrained(self.config.model_dir)

        summary = {
            "model": "florence2",
            "checkpoint": self.config.checkpoint,
            "epochs": num_epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "device": str(self.device),
            "low_vram_mode": self.low_vram,
            "dataset": str(self.config.dataset_yaml),
            "history": history,
            "model_dir": str(self.config.model_dir),
        }
        (self.config.model_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
