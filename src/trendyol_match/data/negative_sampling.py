"""Negatif örnekleme.

Eğitim verisi YALNIZCA alakalı (pozitif) çiftler içerir. Sınıflandırıcı
eğitebilmek için negatif (alakasız) çiftler üretmemiz gerekir.

Strateji — iki tür negatif:
  * Rastgele (kolay): terimi, terimin pozitif olduğu kategoriler DIŞINDAN
    rastgele bir ürünle eşle. Büyük ürün uzayında bunlar neredeyse kesin alakasız.
  * Zor (hard): terimi, taksonomide "yakın" (kategori yolunda ortak kelimesi olan)
    ama pozitif olmayan bir kategoriden bir ürünle eşle. Model ince ayrımları öğrenir.

Kritik ilke: bir negatif çift, terimin herhangi bir pozitif kategorisinden GELMEZ →
böylece üretilen negatiflerin gerçekten alakasız olma olasılığı çok yüksek olur
(etiket gürültüsü minimum).

Ölçek notu: gerçek veri 250K pozitif / ~3K kategori içerir. Kategori adayları
ters indeks (token -> kategoriler) ve terim-bazlı önbellekle O(1)'e yakın seçilir;
ürünler kategori-bazlı indeks listelerinden örneklenir (satır kopyalamadan).

`catalog` verilirse negatif ürünler tüm ürün kataloğundan örneklenir (test,
kataloğun ~%97'sini kapsadığından model yalnız pozitiflerde görülen ürünlere
aşırı uyum sağlamaz); verilmezse pozitif satırlardaki ürünler kullanılır.
"""
from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from ..features.text_normalize import content_tokens


def _category_tokens(category: str) -> set[str]:
    return set(content_tokens(category))


def build_negatives(
    train: pd.DataFrame,
    ratio: float = 1.0,
    random_frac: float = 0.5,
    hard_frac: float = 0.5,
    seed: int = 42,
    catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pozitif çiftlerden negatif çiftler üretir ve pozitiflerle birleştirir.

    Dönüş: term, product_title, category, attributes, label (1=pozitif, 0=negatif)
    kolonlarını içeren dengeli bir DataFrame.
    """
    rng = random.Random(seed)
    pos = train.copy()
    pos["label"] = 1

    # Terim -> pozitif kategorileri (negatif bu kategorilerden GELMEZ)
    term_to_cats: dict[str, set[str]] = defaultdict(set)
    for term, cat in zip(pos["term"], pos["category"]):
        term_to_cats[term].add(cat)

    # Negatif ürün havuzu: tam katalog (varsa) ya da pozitif satırlar
    pool = catalog if catalog is not None else pos
    pool_titles = pool["product_title"].fillna("").astype(str).tolist()
    pool_cats = pool["category"].fillna("").astype(str).tolist()
    pool_attrs = (pool["attributes"].fillna("").astype(str).tolist()
                  if "attributes" in pool.columns else [""] * len(pool))

    # Kategori -> havuzdaki satır indeksleri (ürün örnekleme için)
    cat_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(pool_cats):
        cat_to_idx[c].append(i)
    all_cats = list(cat_to_idx.keys())

    # Token -> kategoriler ters indeksi (zor negatif adayları için)
    every_cat = set(all_cats)
    for cats in term_to_cats.values():
        every_cat |= cats
    cat_token_cache = {c: _category_tokens(c) for c in every_cat}
    token_to_cats: dict[str, set[str]] = defaultdict(set)
    for c in all_cats:
        for t in cat_token_cache[c]:
            token_to_cats[t].add(c)

    def disallowed(term: str) -> set[str]:
        return term_to_cats.get(term, set())

    hard_cache: dict[str, list[str]] = {}

    def pick_hard_category(term: str) -> str | None:
        """Terimin pozitif kategorileriyle ORTAK kelimesi olan ama pozitif
        OLMAYAN bir kategori seç (taksonomik olarak yakın = zor negatif)."""
        cands = hard_cache.get(term)
        if cands is None:
            pos_tokens: set[str] = set()
            for c in disallowed(term):
                pos_tokens |= cat_token_cache.get(c, set())
            cand_set: set[str] = set()
            for t in pos_tokens:
                cand_set |= token_to_cats.get(t, set())
            cands = [c for c in cand_set if c not in disallowed(term)]
            hard_cache[term] = cands
        return rng.choice(cands) if cands else None

    def pick_random_category(term: str) -> str | None:
        """Yasaklı küme küçük olduğundan reddetmeli örnekleme pratikte O(1)."""
        banned = disallowed(term)
        for _ in range(50):
            c = rng.choice(all_cats)
            if c not in banned:
                return c
        cands = [c for c in all_cats if c not in banned]
        return rng.choice(cands) if cands else None

    n_neg = int(len(pos) * ratio)
    n_hard = int(n_neg * hard_frac)
    terms_list = pos["term"].tolist()

    neg_rows = []
    attempts = 0
    max_attempts = n_neg * 20
    while len(neg_rows) < n_neg and attempts < max_attempts:
        attempts += 1
        term = rng.choice(terms_list)
        want_hard = len(neg_rows) < n_hard
        cat = pick_hard_category(term) if want_hard else pick_random_category(term)
        if cat is None:
            cat = pick_random_category(term)  # yedek
        if cat is None or cat not in cat_to_idx:
            continue
        j = rng.choice(cat_to_idx[cat])
        neg_rows.append({
            "term": term,
            "product_title": pool_titles[j],
            "category": pool_cats[j],
            "attributes": pool_attrs[j],
            "label": 0,
        })

    neg = pd.DataFrame(neg_rows)
    keep = ["term", "product_title", "category", "attributes", "label"]
    combined = pd.concat([pos[keep], neg[keep]], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[negatif] pozitif={len(pos)}  negatif={len(neg)} "
          f"(zor~{min(n_hard, len(neg))}, rastgele~{max(0, len(neg) - n_hard)})  "
          f"havuz={'katalog:' + str(len(pool)) if catalog is not None else 'pozitifler'}  "
          f"toplam={len(combined)}")
    return combined
