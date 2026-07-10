"""Değerlendirme metrikleri ve karar eşiği seçimi.

Yarışmanın 'Başarı' ekseni ikili sınıflandırma başarısıyla ölçülür. Test verisi
hem pozitif hem negatif içerdiğinden, olasılık çıktısını sınıfa çeviren KARAR EŞİĞİ
kritik önemdedir. Eşiği F1'i (varsayılan) maksimize edecek şekilde seçeriz.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> tuple[float, float]:
    """PR eğrisi üzerindeki eşikleri tarayarak metriği maksimize eden eşiği bulur.

    Dönüş: (en_iyi_eşik, en_iyi_skor)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds uzunluğu precision/recall'dan 1 eksik; son noktayı atla
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "f1":
            scores = 2 * precision * recall / (precision + recall)
        elif metric == "precision":
            scores = precision
        elif metric == "recall":
            scores = recall
        else:
            raise ValueError(f"Bilinmeyen metrik: {metric}")
    scores = np.nan_to_num(scores, nan=0.0)
    if len(scores) == 0:
        return 0.5, 0.0
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def format_metrics(metrics: dict[str, float]) -> str:
    order = ["f1", "precision", "recall", "accuracy", "roc_auc", "pr_auc", "threshold"]
    parts = [f"{k}={metrics[k]:.4f}" for k in order if k in metrics and metrics[k] == metrics[k]]
    return "  ".join(parts)
