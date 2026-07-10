#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oct_macular.config import load_yaml, resolve_path, save_yaml
from oct_macular.data import validate_manifest
from oct_macular.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_classification_report,
    save_metrics,
)
from oct_macular.models import ManifestImageDataset, build_model, build_transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_names_from_config(config: dict) -> list[str]:
    return list(config.get("data", {}).get("classes", ["CNV", "DME", "DRUSEN", "NORMAL"]))


def make_run_dir(config: dict, run_id: str | None) -> Path:
    output_root = resolve_path(config.get("output", {}).get("dir", "outputs"))
    name = config.get("experiment", {}).get("name", config["model"]["name"])
    resolved_run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}"
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_manifest(config: dict, manifest_override: str | None) -> pd.DataFrame:
    manifest_path = resolve_path(manifest_override or config["data"]["manifest"])
    df = pd.read_csv(manifest_path)
    validate_manifest(df)
    return df


def encode_labels(labels: pd.Series, class_names: list[str]) -> np.ndarray:
    class_to_idx = {label: index for index, label in enumerate(class_names)}
    return labels.map(class_to_idx).to_numpy(dtype=np.int64)


def load_hog_features(rows: pd.DataFrame, config: dict) -> np.ndarray:
    image_size = int(config.get("data", {}).get("image_size", 224))
    hog_config = config.get("model", {}).get("hog", {})
    features = []
    for path in tqdm(rows["image_path"].tolist(), desc="HOG", leave=False):
        with Image.open(path) as image:
            image = image.convert("L").resize((image_size, image_size))
            array = np.asarray(image, dtype=np.float32) / 255.0
        features.append(
            hog(
                array,
                orientations=int(hog_config.get("orientations", 9)),
                pixels_per_cell=tuple(hog_config.get("pixels_per_cell", [16, 16])),
                cells_per_block=tuple(hog_config.get("cells_per_block", [2, 2])),
                feature_vector=True,
            )
        )
    return np.vstack(features)


def save_eval_artifacts(
    run_dir: Path,
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    class_names: list[str],
) -> dict[str, float | None]:
    metrics = compute_metrics(y_true, y_pred, y_score, class_names)
    stem = f"{prefix}_" if prefix else ""
    plot_confusion_matrix(y_true, y_pred, class_names, run_dir / f"{stem}confusion_matrix.png")
    plot_roc_curve(y_true, y_score, class_names, run_dir / f"{stem}roc_curve.png")
    save_classification_report(
        y_true,
        y_pred,
        class_names,
        run_dir / f"{stem}classification_report.txt",
    )
    return metrics


def train_hog_logreg(config: dict, df: pd.DataFrame, run_dir: Path, class_names: list[str]) -> None:
    train_rows = df[df["split"] == "train"]
    val_rows = df[df["split"] == "val"]
    test_rows = df[df["split"] == "test"]
    lr_config = config.get("model", {}).get("logistic_regression", {})
    class_weight = "balanced" if config.get("imbalance", {}).get("enabled", True) else None

    x_train = load_hog_features(train_rows, config)
    y_train = encode_labels(train_rows["label"], class_names)
    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=int(lr_config.get("max_iter", 1000)),
                    solver=lr_config.get("solver", "lbfgs"),
                    class_weight=class_weight,
                    multi_class="auto",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, run_dir / "model.joblib")

    all_metrics = {"classes": class_names, "imbalance": {"strategy": "class_weight"}}
    for split_name, rows in (("val", val_rows), ("test", test_rows)):
        x_split = load_hog_features(rows, config)
        y_true = encode_labels(rows["label"], class_names)
        y_score = pipeline.predict_proba(x_split)
        y_pred = np.argmax(y_score, axis=1)
        all_metrics[split_name] = save_eval_artifacts(
            run_dir,
            "" if split_name == "test" else split_name,
            y_true,
            y_pred,
            y_score,
            class_names,
        )
    save_metrics(all_metrics, run_dir / "metrics.json")


def make_class_weights(rows: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    counts = rows["label"].value_counts().to_dict()
    total = sum(counts.values())
    weights = [
        total / (len(class_names) * max(1, counts.get(class_name, 0)))
        for class_name in class_names
    ]
    return torch.tensor(weights, dtype=torch.float32)


def make_sampler(rows: pd.DataFrame, class_names: list[str]) -> WeightedRandomSampler:
    class_weights = make_class_weights(rows, class_names)
    class_to_idx = {label: index for index, label in enumerate(class_names)}
    sample_weights = [
        float(class_weights[class_to_idx[label]])
        for label in rows["label"].tolist()
    ]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "backbone_frozen", False):
        model.features.eval()
    losses = []
    predictions = []
    targets = []
    for inputs, labels in tqdm(loader, desc="train" if training else "eval", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()) * inputs.size(0))
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        targets.extend(labels.detach().cpu().numpy().tolist())
    return (
        sum(losses) / max(1, len(loader.dataset)),
        np.asarray(targets, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
    )


@torch.no_grad()
def predict_cnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    for inputs, labels in tqdm(loader, desc="predict", leave=False):
        inputs = inputs.to(device)
        logits = model(inputs)
        scores = torch.softmax(logits, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(torch.argmax(scores, dim=1).cpu().numpy().tolist())
        y_score.extend(scores.cpu().numpy().tolist())
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_score, dtype=np.float32),
    )


def train_cnn(config: dict, df: pd.DataFrame, run_dir: Path, class_names: list[str], device: torch.device) -> None:
    train_rows = df[df["split"] == "train"]
    val_rows = df[df["split"] == "val"]
    test_rows = df[df["split"] == "test"]
    image_size = int(config.get("data", {}).get("image_size", 224))
    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 2))
    model_name = config["model"]["name"]
    class_to_idx = {label: index for index, label in enumerate(class_names)}

    imagenet_norm = model_name == "mobilenetv3_small"
    train_dataset = ManifestImageDataset(
        train_rows,
        class_to_idx,
        transform=build_transforms(image_size, train=True, imagenet_norm=imagenet_norm),
    )
    eval_transform = build_transforms(image_size, train=False, imagenet_norm=imagenet_norm)
    val_dataset = ManifestImageDataset(val_rows, class_to_idx, transform=eval_transform)
    test_dataset = ManifestImageDataset(test_rows, class_to_idx, transform=eval_transform)

    strategy = config.get("imbalance", {}).get("strategy", "weighted_sampler")
    imbalance_enabled = config.get("imbalance", {}).get("enabled", True)
    sampler = make_sampler(train_rows, class_names) if imbalance_enabled and strategy in {"weighted_sampler", "both"} else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(
        model_name,
        num_classes=len(class_names),
        pretrained=bool(config.get("model", {}).get("pretrained", False)),
        freeze_backbone=bool(config.get("model", {}).get("freeze_backbone", False)),
    ).to(device)

    class_weight_tensor = None
    if imbalance_enabled and strategy in {"class_weighted_loss", "both"}:
        class_weight_tensor = make_class_weights(train_rows, class_names).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(training_config.get("learning_rate", 0.001)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)

    best_val_loss = float("inf")
    stale_epochs = 0
    patience = int(training_config.get("patience", 5))
    history_rows = []
    checkpoint_path = run_dir / "best_model.pt"
    for epoch in range(1, int(training_config.get("epochs", 20)) + 1):
        train_loss, y_train, pred_train = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_val, pred_val = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        train_balanced = float(compute_metrics(y_train, pred_train, None, class_names)["balanced_accuracy"])
        val_balanced = float(compute_metrics(y_val, pred_val, None, class_names)["balanced_accuracy"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_balanced_accuracy": train_balanced,
                "val_balanced_accuracy": val_balanced,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_val, pred_val, score_val = predict_cnn(model, val_loader, device)
    y_test, pred_test, score_test = predict_cnn(model, test_loader, device)
    all_metrics = {
        "classes": class_names,
        "imbalance": {"strategy": strategy if imbalance_enabled else "none"},
        "val": save_eval_artifacts(run_dir, "val", y_val, pred_val, score_val, class_names),
        "test": save_eval_artifacts(run_dir, "", y_test, pred_test, score_test, class_names),
    }
    save_metrics(all_metrics, run_dir / "metrics.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an OCT classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Override manifest path from config.")
    parser.add_argument("--run-id", help="Optional output run id.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.manifest:
        config.setdefault("data", {})["manifest"] = args.manifest
    set_seed(int(config.get("experiment", {}).get("seed", 42)))
    class_names = class_names_from_config(config)
    df = load_manifest(config, args.manifest)
    run_dir = make_run_dir(config, args.run_id)
    save_yaml(config, run_dir / "config.yaml")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    model_name = config["model"]["name"]
    if model_name == "hog_logreg":
        train_hog_logreg(config, df, run_dir, class_names)
    elif model_name in {"simple_cnn", "mobilenetv3_small"}:
        train_cnn(config, df, run_dir, class_names, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Run written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
