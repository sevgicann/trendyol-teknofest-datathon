"""Keşifsel Veri Analizi (EDA).

Konsola özet istatistikler yazar ve reports/figures/ altına grafikler kaydeder.
Sinyal kontrolü: pozitif çiftlerde terim–başlık sözcük örtüşmesi yüksek olmalı.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # başsız (headless) ortam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import load_config, resolve_path
from ..data.loader import load_test, load_train
from ..features.text_normalize import content_tokens


def _section(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def run_eda(cfg=None) -> None:
    cfg = cfg or load_config()
    fig_dir = resolve_path(cfg, cfg.paths.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train = load_train(cfg)
    test = load_test(cfg)

    _section("1. Boyut & Şema")
    print(f"train: {train.shape}  |  test: {test.shape}")
    print(f"train kolonları: {list(train.columns)}")
    print(f"test  kolonları: {list(test.columns)}")

    _section("2. Eksik Değerler")
    for name, df in [("train", train), ("test", test)]:
        miss = {}
        for col in df.columns:
            s = df[col]
            n_miss = int(s.isna().sum())
            if s.dtype == object:
                n_miss += int((s == "").sum())
            if n_miss:
                miss[col] = n_miss
        print(f"{name}: " + (str(miss) if miss else "eksik değer yok"))

    _section("3. Benzersizlik")
    for col in ["term", "product_title", "category"]:
        if col in train.columns:
            print(f"train.{col}: {train[col].nunique()} benzersiz / {len(train)}")
    if "label" in train.columns:
        print(f"train etiket dağılımı: {train['label'].value_counts().to_dict()} "
              f"(train yalnız pozitif olmalı → negatifler örneklemeyle üretilecek)")

    _section("4. Metin Uzunlukları (kelime sayısı)")
    train_term_wc = train["term"].str.split().str.len()
    train_title_wc = train["product_title"].str.split().str.len()
    print(f"terim kelime sayısı  : ort={train_term_wc.mean():.2f}  medyan={train_term_wc.median():.0f}  max={train_term_wc.max():.0f}")
    print(f"başlık kelime sayısı : ort={train_title_wc.mean():.2f}  medyan={train_title_wc.median():.0f}  max={train_title_wc.max():.0f}")

    _section("5. En sık kategoriler (train)")
    print(train["category"].value_counts().head(10).to_string())

    _section("6. Sinyal Kontrolü — Terim/Başlık sözcük örtüşmesi (pozitif çiftler)")
    overlaps = []
    for term, title in zip(train["term"], train["product_title"]):
        tt, ti = set(content_tokens(term)), set(content_tokens(title))
        overlaps.append((len(tt & ti) / len(tt)) if tt else 0.0)
    overlaps = np.array(overlaps)
    print(f"terim token'larının başlıkta bulunma oranı: ort={overlaps.mean():.3f}  medyan={np.median(overlaps):.3f}")
    print("  -> Pozitiflerde yüksek örtüşme, sözlüksel özniteliklerin güçlü sinyal taşıdığını gösterir.")

    # --- Grafikler ---
    plt.figure(figsize=(7, 4))
    plt.hist(train_term_wc.dropna(), bins=range(0, int(train_term_wc.max()) + 2), alpha=0.7, label="terim")
    plt.hist(train_title_wc.dropna(), bins=range(0, int(train_title_wc.max()) + 2), alpha=0.7, label="başlık")
    plt.xlabel("kelime sayısı"); plt.ylabel("frekans"); plt.legend(); plt.title("Metin uzunluğu dağılımı")
    plt.tight_layout(); plt.savefig(fig_dir / "text_lengths.png", dpi=110); plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(overlaps, bins=20, color="steelblue")
    plt.xlabel("terim→başlık token kapsama oranı"); plt.ylabel("frekans")
    plt.title("Pozitif çiftlerde sözcük örtüşmesi")
    plt.tight_layout(); plt.savefig(fig_dir / "term_title_overlap.png", dpi=110); plt.close()

    top_cats = train["category"].value_counts().head(12)
    plt.figure(figsize=(8, 5))
    top_cats.iloc[::-1].plot(kind="barh", color="darkorange")
    plt.xlabel("adet"); plt.title("En sık kategoriler (train)")
    plt.tight_layout(); plt.savefig(fig_dir / "top_categories.png", dpi=110); plt.close()

    print(f"\n[tamam] Grafikler kaydedildi: {fig_dir}")
