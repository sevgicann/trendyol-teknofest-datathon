"""Eğitim boru hattı.

Adımlar:
  1) Pozitif (alakalı) train verisini yükle
  2) Negatif örnekleme ile dengeli eğitim seti kur
  3) Öznitelikleri fit + transform et
  4) Terim-bazlı GroupKFold ile LightGBM eğit (OOF tahminleri)
  5) OOF üzerinde F1'i maksimize eden karar eşiğini seç
  6) Model + öznitelik oluşturucu + eşiği kaydet; öznitelik önemlerini raporla
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_config, resolve_path
from ..data.loader import load_items, load_train
from ..data.negative_sampling import build_negatives
from ..evaluation.metrics import evaluate, find_best_threshold, format_metrics
from ..features.build_features import FeatureBuilder
from ..models.lgbm_model import LgbmMatcher


def run_train(cfg=None) -> dict:
    cfg = cfg or load_config()
    models_dir = resolve_path(cfg, cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = resolve_path(cfg, cfg.paths.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1) pozitif veri
    train_pos = load_train(cfg)

    # 2) negatif örnekleme — negatif havuzu olarak tam ürün kataloğu (varsa)
    try:
        catalog = load_items(cfg)
    except FileNotFoundError:
        catalog = None  # düz (flat) şema: negatifler pozitif satırlardan örneklenir
    ns = cfg.negative_sampling
    data = build_negatives(
        train_pos,
        ratio=ns["ratio"], random_frac=ns["random_frac"],
        hard_frac=ns["hard_frac"], seed=ns["seed"],
        catalog=catalog,
    )
    data.to_csv(interim_dir / "train_with_negatives.csv", index=False, encoding="utf-8")

    # 3) öznitelikler
    print("\n[öznitelik] TF-IDF fit + transform...")
    fb = FeatureBuilder(cfg)
    X = fb.fit_transform(data)
    y = data["label"].values
    groups = data["term"].values
    print(f"[öznitelik] {X.shape[1]} öznitelik: {list(X.columns)}")

    # 4) model
    print("\n[model] LightGBM GroupKFold CV...")
    model = LgbmMatcher(cfg)
    model.fit(X, y, groups=groups, verbose=True)

    # 5) eşik seçimi (OOF üzerinde)
    metric = cfg.evaluation.optimize_metric
    best_thr, best_score = find_best_threshold(y, model.oof_, metric=metric)
    oof_metrics = evaluate(y, model.oof_, threshold=best_thr)
    print("\n[OOF] en iyi eşik ve skorlar:")
    print(f"  eşik={best_thr:.4f}  ({metric}={best_score:.4f})")
    print("  " + format_metrics(oof_metrics))

    # 6) kaydet
    fb.save(models_dir / "feature_builder.pkl")
    model.save(models_dir / "lgbm_matcher.pkl")
    meta = {
        "threshold": best_thr,
        "optimize_metric": metric,
        "oof_metrics": oof_metrics,
        "n_features": int(X.shape[1]),
        "n_train": int(len(data)),
    }
    with open(models_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # açıklanabilirlik: öznitelik önemleri
    fi = model.feature_importance_
    fi.to_csv(models_dir / "feature_importance.csv", index=False, encoding="utf-8")
    print("\n[açıklanabilirlik] Öznitelik önemleri (ilk 10):")
    print(fi.head(10).to_string(index=False))

    print(f"\n[tamam] Artefaktlar kaydedildi: {models_dir}")
    print("  Sıradaki: python scripts/04_predict.py")
    return meta


if __name__ == "__main__":
    run_train()
