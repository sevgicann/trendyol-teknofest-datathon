"""Pseudo-labeling — modellerin MUTABIK ve EMİN olduğu test çiftlerini eğitime katar.

Temel sorun: eğitimde gerçek test negatiflerini hiç görmüyoruz (train pozitif-only,
sentetik negatifler gerçek dağılımı temsil etmiyor — v4 dersi). Güvenli çözüm:
birden fazla modelin AYNI ANDA çok emin olduğu test çiftlerini o etiketle eğitime
almak. Modeller emin oldukları uçlarda ~%97+ isabetlidir; iki-model mutabakatı +
muhafazakâr eşikler etiket gürültüsünü küçük tutar. Böylece model ilk kez GERÇEK
test dağılımından (özellikle gerçek negatiflerden) öğrenir.

Not: v4'ün hatasıyla karıştırılmamalı — orada 'benzer=negatif' varsayımıyla
etiket UYDURuyorduk; burada modelin kendi yüksek-güvenli tahminini kullanıyoruz.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd


def build_pseudo_labels(
    test: pd.DataFrame,
    proba_paths: list,
    pos_thr: float = 0.90,
    neg_thr: float = 0.03,
    max_pos: int = 100_000,
    max_neg: int = 300_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test çiftlerinden yüksek-güvenli pseudo-etiketli eğitim satırları üretir.

    test: load_test çıktısı (metin kolonları join edilmiş, satır sırası olasılık
    dosyalarıyla aynı). proba_paths: her biri (id, probability) CSV'si; TÜM
    modeller pos_thr üstündeyse pozitif, TÜMÜ neg_thr altındaysa negatif.
    """
    rng = random.Random(seed)
    probas = []
    for p in proba_paths:
        df = pd.read_csv(p)
        if not (df["id"].values == test["id"].values).all():
            raise ValueError(f"{p}: id sırası test ile uyuşmuyor")
        probas.append(df["probability"].values)
    P = np.vstack(probas)

    pos_mask = (P > pos_thr).all(axis=0)
    neg_mask = (P < neg_thr).all(axis=0)
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(neg_mask)
    if len(pos_idx) > max_pos:
        pos_idx = np.array(rng.sample(list(pos_idx), max_pos))
    if len(neg_idx) > max_neg:
        neg_idx = np.array(rng.sample(list(neg_idx), max_neg))

    cols = [c for c in ["term", "term_id", "item_id", "product_title", "category",
                        "attributes", "brand", "gender", "age_group"] if c in test.columns]
    pseudo_pos = test.iloc[pos_idx][cols].copy(); pseudo_pos["label"] = 1
    pseudo_neg = test.iloc[neg_idx][cols].copy(); pseudo_neg["label"] = 0
    out = pd.concat([pseudo_pos, pseudo_neg], ignore_index=True)
    print(f"[pseudo] aday: +{int(pos_mask.sum())} / -{int(neg_mask.sum())}  "
          f"alınan: +{len(pseudo_pos)} / -{len(pseudo_neg)}  "
          f"(eşikler: >{pos_thr} / <{neg_thr}, {len(proba_paths)} model mutabakatı)")
    return out
