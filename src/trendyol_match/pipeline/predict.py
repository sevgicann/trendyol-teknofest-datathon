"""Çıkarım boru hattı.

Kaydedilmiş öznitelik oluşturucu + model + eşik ile test verisini puanlar ve
Kaggle submission dosyası üretir. Sentetik veride gerçek etiket varsa yerel
skoru da raporlar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..data.loader import load_sample_submission, load_test, load_test_labels
from ..evaluation.metrics import evaluate, format_metrics
from ..features.build_features import FeatureBuilder
from ..models.lgbm_model import LgbmMatcher


def run_predict(cfg=None) -> Path:
    cfg = cfg or load_config()
    models_dir = resolve_path(cfg, cfg.paths.models_dir)
    sub_dir = resolve_path(cfg, cfg.paths.submissions_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)

    # artefaktlar
    fb = FeatureBuilder.load(models_dir / "feature_builder.pkl")
    model = LgbmMatcher.load(models_dir / "lgbm_matcher.pkl")
    with open(models_dir / "train_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["threshold"])

    # test — büyük test setleri (milyonlarca çift) bellek dostu parçalar halinde puanlanır
    test = load_test(cfg)
    chunk_size = int(cfg.get("inference", {}).get("chunk_size", 250_000))
    n = len(test)
    proba = np.empty(n, dtype=np.float64)
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, n, chunk_size), 1):
        stop = min(start + chunk_size, n)
        part = test.iloc[start:stop]
        X_part = fb.transform(part)
        proba[start:stop] = model.predict_proba(X_part)
        if n_chunks > 1:
            print(f"  [çıkarım] parça {ci}/{n_chunks}  ({stop}/{n} satır)")
    pred = (proba >= threshold).astype(int)

    # submission formatını sample_submission'a uydur
    sample = load_sample_submission(cfg)
    id_col = cfg.submission.id_column
    pred_col = cfg.submission.prediction_column
    if not sample.empty:
        cols = list(sample.columns)
        id_col = cols[0]
        pred_col = cols[1] if len(cols) > 1 else pred_col

    submission = pd.DataFrame({id_col: test["id"].values, pred_col: pred})
    out_path = sub_dir / "submission.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[submission] {out_path}  ({len(submission)} satır)")
    print(f"  eşik={threshold:.4f}  pozitif tahmin oranı={pred.mean():.3f}")
    print(submission.head().to_string(index=False))

    # olasılıkları da sakla (analiz / ansambl için)
    pd.DataFrame({id_col: test["id"].values, "probability": proba}).to_csv(
        sub_dir / "test_probabilities.csv", index=False, encoding="utf-8")

    # yerel skor (yalnız sentetik veride gerçek etiket varsa VE id'ler eşleşiyorsa)
    labels = load_test_labels(cfg)
    if labels is not None:
        merged = test[["id"]].merge(labels, on="id", how="left")
        mask = merged["label"].notna().values
        if mask.any():
            y_true = merged.loc[mask, "label"].values
            m = evaluate(y_true, proba[mask], threshold=threshold)
            print("\n[yerel skor] (test etiketlerine karşı)")
            print("  " + format_metrics(m))

    return out_path


if __name__ == "__main__":
    run_predict()
