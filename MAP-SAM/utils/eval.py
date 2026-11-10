"""Semantic-segmentation metrics and validation helpers."""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from config import METRIC_POLICY_VERSION
from utils.reproducibility import fixed_torch_seed


DEFAULT_LABEL_PATH = os.path.join(os.path.dirname(__file__), "classes.json")


class ConfusionMatrix:
    """Confusion matrix with rows as targets and columns as predictions."""

    def __init__(self, num_classes: int, labels: Sequence[str] | None = None) -> None:
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes
        self.labels = list(labels) if labels is not None else [str(i) for i in range(num_classes)]
        if len(self.labels) != num_classes:
            raise ValueError("labels must contain one name per class")

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.int64)
        target = np.asarray(target, dtype=np.int64)
        valid = (
            (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        indices = self.num_classes * target[valid] + prediction[valid]
        self.matrix += np.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def metrics(
        self, class_ids: Iterable[int]
    ) -> Tuple[List[Dict[str, float]], float, float, float]:
        class_ids = list(class_ids)
        rows: List[Dict[str, float]] = []
        for class_id in class_ids:
            true_positive = self.matrix[class_id, class_id]
            false_positive = self.matrix[:, class_id].sum() - true_positive
            false_negative = self.matrix[class_id, :].sum() - true_positive
            denominator = true_positive + false_positive + false_negative
            iou = true_positive / denominator if denominator else float("nan")
            f1_denominator = 2 * true_positive + false_positive + false_negative
            f1 = (
                2 * true_positive / f1_denominator
                if f1_denominator
                else float("nan")
            )
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": self.labels[class_id],
                    "iou": iou * 100.0,
                    "f1": f1 * 100.0,
                }
            )

        valid_targets = self.matrix.sum()
        correct = np.trace(self.matrix)
        overall_accuracy = correct / valid_targets * 100.0 if valid_targets else 0.0
        valid_iou = [row["iou"] for row in rows if np.isfinite(row["iou"])]
        valid_f1 = [row["f1"] for row in rows if np.isfinite(row["f1"])]
        mean_iou = float(np.mean(valid_iou)) if valid_iou else 0.0
        mean_f1 = float(np.mean(valid_f1)) if valid_f1 else 0.0
        return rows, mean_iou, mean_f1, overall_accuracy

    def summary(self, class_ids: Iterable[int]) -> Tuple[float, float, float]:
        rows, mean_iou, mean_f1, overall_accuracy = self.metrics(class_ids)
        table = "Per-class | IoU (%) | F1-score (%)\n" + "\n".join(
            f"{row['class_name']} | {row['iou']:.3f} | {row['f1']:.3f}"
            for row in rows
        )
        logging.info("Classes acc:\n%s", table)
        logging.info("\tmIoU: %.3f %%", mean_iou)
        logging.info("\tMacro F1-score: %.3f %%", mean_f1)
        logging.info("\tOA: %.3f %%", overall_accuracy)
        return mean_iou, mean_f1, overall_accuracy

    def plot(self, save_path: str, class_ids: Iterable[int]) -> None:
        import matplotlib.pyplot as plt

        class_ids = list(class_ids)
        matrix = self.matrix[np.ix_(class_ids, class_ids)].astype(np.float64)
        matrix /= matrix.sum(axis=1, keepdims=True) + 1.0e-8
        labels = [self.labels[class_id] for class_id in class_ids]
        plt.figure(figsize=(12, 9), tight_layout=True)
        try:
            import seaborn as sns

            sns.set(font_scale=1.0 if len(class_ids) < 50 else 0.8)
            sns.heatmap(
                data=matrix,
                annot=len(class_ids) < 30,
                annot_kws={"size": 12},
                cmap="Blues",
                fmt=".3f",
                square=True,
                linewidths=0.16,
                linecolor="white",
                xticklabels=labels,
                yticklabels=labels,
            )
        except ImportError:
            plt.imshow(matrix, cmap="Blues")
            plt.xticks(range(len(class_ids)), labels, rotation=45)
            plt.yticks(range(len(class_ids)), labels)
            plt.colorbar()

        plt.xlabel("Predicted", size=16, color="purple")
        plt.ylabel("True", size=16, color="purple")
        plt.title("Normalized confusion matrix", size=20, color="blue")
        plt.savefig(save_path + ".png", dpi=150)
        plt.close()

        with open(save_path + ".csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([""] + self.labels)
            for label, row in zip(self.labels, self.matrix):
                writer.writerow([label] + list(row))


def load_class_names(json_label_path: str = DEFAULT_LABEL_PATH) -> List[str]:
    if not os.path.exists(json_label_path):
        raise FileNotFoundError(f"Cannot find {json_label_path} file")
    with open(json_label_path, "r", encoding="utf-8") as file:
        class_indict = json.load(file)
    return list(class_indict)


def metric_class_ids(num_classes: int, include_background: bool) -> range:
    return range(num_classes) if include_background else range(num_classes - 1)


def eval_net(
    net,
    loader,
    device,
    bg,
    json_label_path=DEFAULT_LABEL_PATH,
    noise_seed=42,
):
    labels = load_class_names(json_label_path)
    confusion = ConfusionMatrix(num_classes=len(labels), labels=labels)
    class_ids = metric_class_ids(len(labels), include_background=bg)
    net.eval()
    logging.info(
        "Metric policy v%d: background pixels retained; macro average classes=%s",
        METRIC_POLICY_VERSION,
        list(class_ids),
    )

    with fixed_torch_seed(noise_seed, device), torch.inference_mode():
        for batch in loader:
            image = batch["img"].to(device)
            dsm = batch["dsm"].to(device)
            target = batch["mask"].numpy().squeeze(1)
            logits, _, _, _, _ = net(image, dsm, is_train=False)
            prediction = logits.argmax(dim=1).cpu().numpy()
            confusion.update(prediction, target)

    mean_iou, mean_f1, overall_accuracy = confusion.summary(class_ids)
    return mean_iou, mean_f1, overall_accuracy, confusion, len(class_ids)


def eval_net_save(
    net,
    loader,
    device,
    bg,
    json_label_path=DEFAULT_LABEL_PATH,
    save_dir=None,
    noise_seed=42,
):
    from PIL import Image

    labels = load_class_names(json_label_path)
    confusion = ConfusionMatrix(num_classes=len(labels), labels=labels)
    class_ids = metric_class_ids(len(labels), include_background=bg)
    net.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    save_counter = 0
    with fixed_torch_seed(noise_seed, device), torch.inference_mode():
        for batch in loader:
            image = batch["img"].to(device)
            dsm = batch["dsm"].to(device)
            target = batch["mask"].numpy().squeeze(1)
            logits, _, _, _, _ = net(image, dsm, is_train=False)
            prediction = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            confusion.update(prediction, target)

            if save_dir:
                for pred in prediction:
                    save_path = os.path.join(save_dir, f"{save_counter:06d}.png")
                    Image.fromarray(pred).save(save_path)
                    save_counter += 1

    mean_iou, mean_f1, overall_accuracy = confusion.summary(class_ids)
    return mean_iou, mean_f1, overall_accuracy, confusion, len(class_ids)
