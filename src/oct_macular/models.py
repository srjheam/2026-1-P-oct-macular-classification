from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


class ManifestImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, class_to_idx: dict[str, int], transform=None):
        self.rows = rows.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)


def build_transforms(image_size: int, train: bool, imagenet_norm: bool = True):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
            ]
        )
    steps.append(transforms.ToTensor())
    if imagenet_norm:
        steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        steps.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(steps)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name == "mobilenetv3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for parameter in model.features.parameters():
                parameter.requires_grad = False
            model.features.eval()
        model.backbone_frozen = freeze_backbone
        return model
    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    model = build_model(
        config["model"]["name"],
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=bool(config.get("model", {}).get("freeze_backbone", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
