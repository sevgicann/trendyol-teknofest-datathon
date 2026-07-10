"""Sentetik Trendyol-benzeri veri üreteci.

Amaç: gerçek Kaggle verisi inmeden ÖNCE tüm boru hattını uçtan uca çalıştırıp
doğrulayabilmek. Üretilen dosyalar yarışmanın beklenen şemasını taklit eder:

  train.csv              -> SADECE alakalı (pozitif) çiftler (label=1)
  test.csv               -> id + çift (label yok); alakalı ve alakasız karışık
  sample_submission.csv  -> id, prediction
  test_labels.csv        -> (sadece sentetikte) gerçek etiketler; yerel skor için

Gerçek veri geldiğinde bu dosyaların üzerine yazılır ve pipeline aynen çalışır.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from ..config import load_config, resolve_path

# --- Küçük ama gerçekçi Türkçe e-ticaret sözlüğü ---------------------------
CATALOG = {
    "Giyim > Kadın > Elbise": {
        "titles": [
            "Kadın Çiçek Desenli Yazlık Elbise",
            "Kadın Siyah Uzun Kollu Triko Elbise",
            "Kadın Saten Kısa Abiye Elbise",
            "Kadın Kolsuz Günlük Pamuklu Elbise",
        ],
        "terms": ["yazlık elbise", "abiye elbise", "kadın elbise", "çiçekli elbise", "triko elbise"],
        "attrs": ["materyal: pamuk, renk: kırmızı, stil: günlük",
                  "materyal: saten, renk: siyah, stil: abiye",
                  "materyal: triko, renk: lacivert, stil: kışlık"],
    },
    "Ayakkabı > Erkek > Spor Ayakkabı": {
        "titles": [
            "Erkek Beyaz Sneaker Spor Ayakkabı",
            "Erkek Siyah Koşu Ayakkabısı",
            "Erkek Deri Günlük Spor Ayakkabı",
            "Erkek Yürüyüş Ayakkabısı Hafif Taban",
        ],
        "terms": ["erkek spor ayakkabı", "beyaz sneaker", "koşu ayakkabısı", "erkek sneaker", "yürüyüş ayakkabısı"],
        "attrs": ["materyal: deri, renk: beyaz, stil: spor",
                  "materyal: tekstil, renk: siyah, stil: koşu",
                  "materyal: süet, renk: kahverengi, stil: günlük"],
    },
    "Elektronik > Telefon > Cep Telefonu": {
        "titles": [
            "Akıllı Telefon 128 GB Siyah",
            "5G Cep Telefonu 256 GB Mavi",
            "Katlanabilir Ekran Akıllı Telefon",
            "Uygun Fiyatlı Android Telefon 64 GB",
        ],
        "terms": ["akıllı telefon", "cep telefonu", "5g telefon", "android telefon", "128 gb telefon"],
        "attrs": ["renk: siyah, hafıza: 128 gb, ekran: 6.5 inç",
                  "renk: mavi, hafıza: 256 gb, ekran: 6.7 inç",
                  "renk: yeşil, hafıza: 64 gb, ekran: 6.1 inç"],
    },
    "Ev & Yaşam > Mutfak > Tencere": {
        "titles": [
            "Granit Kaplama 7 Parça Tencere Seti",
            "Çelik Düdüklü Tencere 5 Litre",
            "Döküm Tava ve Tencere Takımı",
            "Yapışmaz Granit Tencere Seti",
        ],
        "terms": ["tencere seti", "granit tencere", "düdüklü tencere", "çelik tencere", "tava seti"],
        "attrs": ["materyal: granit, parça: 7, renk: antrasit",
                  "materyal: çelik, hacim: 5 lt, tip: düdüklü",
                  "materyal: döküm, renk: kırmızı, tip: tava"],
    },
    "Kozmetik > Cilt Bakımı > Nemlendirici": {
        "titles": [
            "Hyaluronik Asitli Nemlendirici Krem 50 ml",
            "SPF 30 Güneş Koruyucu Nemlendirici",
            "Gece Onarıcı Yüz Kremi Retinol",
            "Yağlı Ciltler İçin Nemlendirici Jel",
        ],
        "terms": ["nemlendirici krem", "yüz kremi", "hyaluronik asit", "güneş kremi", "gece kremi"],
        "attrs": ["hacim: 50 ml, cilt: kuru, içerik: hyaluronik asit",
                  "hacim: 40 ml, cilt: karma, spf: 30",
                  "hacim: 30 ml, cilt: yağlı, içerik: retinol"],
    },
}


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def generate(
    n_positive: int = 4000,
    n_test: int = 1500,
    test_positive_frac: float = 0.5,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = _rng(seed)
    cats = list(CATALOG.keys())

    def sample_pair_related(cat: str):
        c = CATALOG[cat]
        return (rng.choice(c["terms"]), rng.choice(c["titles"]), cat, rng.choice(c["attrs"]))

    def sample_pair_unrelated():
        cat_t, cat_p = rng.sample(cats, 2)  # farklı kategoriler
        term = rng.choice(CATALOG[cat_t]["terms"])
        title = rng.choice(CATALOG[cat_p]["titles"])
        attrs = rng.choice(CATALOG[cat_p]["attrs"])
        return (term, title, cat_p, attrs)

    # --- TRAIN: yalnız pozitif çiftler ---
    train_rows = []
    for _ in range(n_positive):
        cat = rng.choice(cats)
        term, title, category, attrs = sample_pair_related(cat)
        train_rows.append((term, title, category, attrs, 1))
    train = pd.DataFrame(train_rows, columns=["search_term", "product_title", "category", "attributes", "label"])

    # --- TEST: alakalı + alakasız karışık, id'li, label gizli ---
    test_rows = []
    labels = []
    n_pos_test = int(n_test * test_positive_frac)
    for i in range(n_test):
        if i < n_pos_test:
            cat = rng.choice(cats)
            term, title, category, attrs = sample_pair_related(cat)
            y = 1
        else:
            term, title, category, attrs = sample_pair_unrelated()
            y = 0
        test_rows.append((f"test_{i:06d}", term, title, category, attrs))
        labels.append(y)
    test = pd.DataFrame(test_rows, columns=["id", "search_term", "product_title", "category", "attributes"])
    # Test satırlarını karıştır (sıralama sinyali olmasın)
    order = list(range(n_test))
    rng.shuffle(order)
    test = test.iloc[order].reset_index(drop=True)
    test_labels = pd.DataFrame({"id": test["id"].values, "label": [labels[i] for i in order]})

    sample_submission = pd.DataFrame({"id": test["id"].values, "prediction": 0})

    return {
        "train": train,
        "test": test,
        "sample_submission": sample_submission,
        "test_labels": test_labels,
    }


def write_synthetic(cfg=None, **kwargs) -> None:
    cfg = cfg or load_config()
    raw_dir = resolve_path(cfg, cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    data = generate(**kwargs)
    for name, df in data.items():
        out = raw_dir / f"{name}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[yaz] {out}  ({len(df)} satır, {df.shape[1]} kolon)")
    print("\n[tamam] Sentetik veri hazır. Şimdi: python scripts/02_eda.py")


if __name__ == "__main__":
    write_synthetic()
