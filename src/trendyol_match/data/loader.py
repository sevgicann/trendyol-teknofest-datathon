"""Şemaya uyarlanabilir veri yükleyici.

Gerçek dosyanın kolon adları config'teki kanonik adlardan farklı olsa bile,
`column_aliases` listesini kullanarak kolonları otomatik eşler. Böylece gerçek
Kaggle verisi indiğinde kodu değiştirmeye gerek kalmaz.

İlişkisel şema desteği: gerçek yarışma verisinde çift dosyaları yalnızca
`term_id` / `item_id` taşır; terim ve ürün metinleri ayrı dosyalardadır
(terms.csv, items.csv). Loader bunu otomatik algılar ve join eder — pipeline'ın
geri kalanı düz (flat) şemayla çalışmaya devam eder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import Config, load_config, resolve_path

# Kod tabanının kullandığı kanonik kolon adları
CANON = ["id", "term", "product_title", "category", "attributes", "label"]


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Veri bulunamadı: {path}\n"
            "  -> Gerçek veri için: python scripts/00_download_data.py\n"
            "  -> Sentetik veri için: python scripts/01_make_synthetic.py"
        )
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8")


def _resolve_column_map(df_cols: list[str], cfg: Config) -> dict[str, Optional[str]]:
    """Kanonik ad -> gerçek dosyadaki kolon adı eşlemesi kurar.

    Öncelik: (1) config.columns'ta belirtilen ad birebir varsa onu kullan,
    (2) yoksa column_aliases içinden küçük-harf eşleşmesi ara.
    """
    lower_to_actual = {c.lower().strip(): c for c in df_cols}
    columns_cfg = cfg.columns
    aliases_cfg = cfg.column_aliases
    mapping: dict[str, Optional[str]] = {}

    for canon in CANON:
        # 1) config'te açıkça verilen isim dosyada var mı?
        explicit = columns_cfg.get(canon)
        if explicit and explicit in df_cols:
            mapping[canon] = explicit
            continue
        if explicit and explicit.lower().strip() in lower_to_actual:
            mapping[canon] = lower_to_actual[explicit.lower().strip()]
            continue
        # 2) alias listesinden ara
        found = None
        for alias in aliases_cfg.get(canon, []):
            if alias.lower().strip() in lower_to_actual:
                found = lower_to_actual[alias.lower().strip()]
                break
        mapping[canon] = found
    return mapping


def _apply_map(df: pd.DataFrame, mapping: dict[str, Optional[str]]) -> pd.DataFrame:
    rename = {actual: canon for canon, actual in mapping.items() if actual is not None}
    out = df.rename(columns=rename)
    # Kanonik kolonları + eşleşmeyen ekstra kolonları koru
    keep = [c for c in CANON if c in out.columns]
    extra = [c for c in out.columns if c not in CANON and c not in rename.values()]
    return out[keep + extra]


# --- İlişkisel şema: terms.csv / items.csv (süreç içi önbellekli) ----------
_LOOKUP_CACHE: dict[str, pd.DataFrame] = {}


def load_terms(cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    """term_id -> term (arama terimi metni) sözlük tablosu."""
    cfg = cfg or load_config()
    rel = cfg.paths.get("terms_file")
    if not rel:
        raise FileNotFoundError("config.paths.terms_file tanımlı değil (ilişkisel şema yok)")
    path = resolve_path(cfg, rel)
    key = str(path)
    if key in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[key]
    df = _read_any(path)
    mapping = _resolve_column_map(list(df.columns), cfg)
    df = _apply_map(df, mapping)
    if verbose:
        print(f"[yükle] terms: {path.name}  ({len(df)} terim)")
    _LOOKUP_CACHE[key] = df
    return df


def load_items(cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    """item_id -> ürün metinleri sözlük tablosu.

    brand / gender / age_group gibi ek alanlar attributes metnine katılır;
    böylece mevcut öznitelikler (TF-IDF, kapsama) bu sinyalleri de görür.
    """
    cfg = cfg or load_config()
    rel = cfg.paths.get("items_file")
    if not rel:
        raise FileNotFoundError("config.paths.items_file tanımlı değil (ilişkisel şema yok)")
    path = resolve_path(cfg, rel)
    key = str(path)
    if key in _LOOKUP_CACHE:
        return _LOOKUP_CACHE[key]
    df = _read_any(path)
    mapping = _resolve_column_map(list(df.columns), cfg)
    df = _apply_map(df, mapping)
    df = _enrich_attributes(df)
    if verbose:
        print(f"[yükle] items: {path.name}  ({len(df)} ürün)")
    _LOOKUP_CACHE[key] = df
    return df


_ATTR_EXTRA = [("brand", "marka"), ("gender", "cinsiyet"), ("age_group", "yaş grubu")]


def _enrich_attributes(items: pd.DataFrame) -> pd.DataFrame:
    """brand/gender/age_group alanlarını attributes metninin başına ekler."""
    extras = [c for c, _ in _ATTR_EXTRA if c in items.columns]
    if not extras:
        return items
    attrs = items["attributes"].fillna("").astype(str) if "attributes" in items.columns \
        else pd.Series([""] * len(items))
    parts = []
    cols = {c: items[c].fillna("").astype(str).tolist() for c in extras}
    for i, base in enumerate(attrs.tolist()):
        bits = [f"{tr}: {cols[c][i]}" for c, tr in _ATTR_EXTRA
                if c in cols and cols[c][i] and cols[c][i] != "unknown"]
        if base:
            bits.append(base)
        parts.append(", ".join(bits))
    items = items.drop(columns=extras)
    items["attributes"] = parts
    return items


def _maybe_join_relational(df: pd.DataFrame, cfg: Config, verbose: bool) -> pd.DataFrame:
    """Çift dosyası yalnız ID taşıyorsa terim/ürün metinlerini join eder."""
    if "term" not in df.columns and "term_id" in df.columns:
        df = df.merge(load_terms(cfg, verbose=verbose), on="term_id", how="left")
    if "product_title" not in df.columns and "item_id" in df.columns:
        df = df.merge(load_items(cfg, verbose=verbose), on="item_id", how="left")
    return df


def load_train(cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_path(cfg, cfg.paths.train_file)
    df = _read_any(path)
    mapping = _resolve_column_map(list(df.columns), cfg)
    if verbose:
        print(f"[yükle] train: {path.name}  ({len(df)} satır)")
        print(f"        kolon eşlemesi: { {k: v for k, v in mapping.items() if v} }")
    df = _apply_map(df, mapping)
    df = _maybe_join_relational(df, cfg, verbose)
    # Train yalnız pozitif çiftler içeriyorsa label yoksa 1 ata
    if "label" not in df.columns:
        df["label"] = 1
    _fill_text_columns(df)
    return df


def load_test(cfg: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_path(cfg, cfg.paths.test_file)
    df = _read_any(path)
    mapping = _resolve_column_map(list(df.columns), cfg)
    if verbose:
        print(f"[yükle] test: {path.name}  ({len(df)} satır)")
    df = _apply_map(df, mapping)
    df = _maybe_join_relational(df, cfg, verbose)
    if "id" not in df.columns:
        df.insert(0, "id", [f"row_{i}" for i in range(len(df))])
    _fill_text_columns(df)
    return df


def load_sample_submission(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_path(cfg, cfg.paths.sample_submission_file)
    if not path.exists():
        return pd.DataFrame()
    return _read_any(path)


def load_test_labels(cfg: Config | None = None) -> Optional[pd.DataFrame]:
    """Sadece sentetik veride bulunur; yerel skorlama için kullanılır."""
    cfg = cfg or load_config()
    path = resolve_path(cfg, cfg.paths.raw_dir) / "test_labels.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8")
    return None


TEXT_COLS = ["term", "product_title", "category", "attributes"]


def _fill_text_columns(df: pd.DataFrame) -> None:
    """Eksik metin kolonlarını boş string ile doldur, NaN'ları temizle."""
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
