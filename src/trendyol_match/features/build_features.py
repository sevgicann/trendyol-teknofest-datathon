"""Öznitelik mühendisliği.

Terim–ürün çiftinden yorumlanabilir sayısal öznitelikler üretir:
  * Sözlüksel örtüşme (Jaccard, overlap katsayısı, ortak token, kapsama)
  * TF-IDF kosinüs benzerliği (kelime n-gram + karakter n-gram)
  * Kategori / nitelik eşleşme sinyalleri
  * Uzunluk / sayım öznitelikleri
  * (Opsiyonel) çok dilli cümle-embedding kosinüs benzerliği

FeatureBuilder durumludur (stateful): TF-IDF sözlüğü train'de FIT edilir, test'te
aynı sözlükle TRANSFORM edilir. Böylece train/test öznitelikleri birebir tutarlı olur.
Açıklanabilirlik için tüm öznitelik adları anlamlıdır.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_normalize import content_tokens, normalize


def _row_cosine(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    """İki L2-normalize TF-IDF matrisi arasında satır-bazlı kosinüs benzerliği."""
    # TfidfVectorizer varsayılan norm='l2' → satır normları 1; kosinüs = satır iç çarpımı
    prod = a.multiply(b)
    return np.asarray(prod.sum(axis=1)).ravel()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class FeatureBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        fcfg = cfg.features
        self.word_vec = TfidfVectorizer(
            max_features=fcfg.tfidf_word["max_features"],
            ngram_range=tuple(fcfg.tfidf_word["ngram_range"]),
            min_df=fcfg.tfidf_word["min_df"],
            lowercase=False,  # normalize() zaten hallediyor
        )
        self.char_vec = TfidfVectorizer(
            analyzer=fcfg.tfidf_char.get("analyzer", "char_wb"),
            max_features=fcfg.tfidf_char["max_features"],
            ngram_range=tuple(fcfg.tfidf_char["ngram_range"]),
            min_df=fcfg.tfidf_char["min_df"],
            lowercase=False,
        )
        self.use_embeddings = bool(fcfg.get("use_embeddings", False))
        self.embedding_model_name = fcfg.get("embedding_model")
        self._embedder = None
        self.feature_names_: list[str] = []

    # --- metin alanları ---
    @staticmethod
    def _prep(df: pd.DataFrame) -> dict[str, list[str]]:
        term = [normalize(t) for t in df["term"].astype(str)]
        title = [normalize(t) for t in df["product_title"].astype(str)]
        category = [normalize(t) for t in df["category"].astype(str)]
        attrs = [normalize(t) for t in df["attributes"].astype(str)]
        product_doc = [f"{ti} {ca} {at}".strip() for ti, ca, at in zip(title, category, attrs)]
        return {"term": term, "title": title, "category": category,
                "attrs": attrs, "product_doc": product_doc}

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        prep = self._prep(df)
        corpus = prep["term"] + prep["product_doc"]
        self.word_vec.fit(corpus)
        self.char_vec.fit(corpus)
        if self.use_embeddings:
            self._load_embedder()
        return self

    def _load_embedder(self):
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            print(f"[embed] model yüklendi: {self.embedding_model_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[embed] embedding devre dışı ({exc}). TF-IDF ile devam.")
            self.use_embeddings = False
            self._embedder = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        prep = self._prep(df)
        n = len(df)

        # --- TF-IDF kosinüs benzerlikleri ---
        term_w = self.word_vec.transform(prep["term"])
        doc_w = self.word_vec.transform(prep["product_doc"])
        title_w = self.word_vec.transform(prep["title"])
        term_c = self.char_vec.transform(prep["term"])
        doc_c = self.char_vec.transform(prep["product_doc"])
        title_c = self.char_vec.transform(prep["title"])

        feats: dict[str, np.ndarray] = {}
        feats["tfidf_word_cos_term_doc"] = _row_cosine(term_w, doc_w)
        feats["tfidf_word_cos_term_title"] = _row_cosine(term_w, title_w)
        feats["tfidf_char_cos_term_doc"] = _row_cosine(term_c, doc_c)
        feats["tfidf_char_cos_term_title"] = _row_cosine(term_c, title_c)

        # --- Sözlüksel örtüşme öznitelikleri ---
        term_tok = [set(content_tokens(t)) for t in df["term"].astype(str)]
        title_tok = [set(content_tokens(t)) for t in df["product_title"].astype(str)]
        cat_tok = [set(content_tokens(t)) for t in df["category"].astype(str)]
        attr_tok = [set(content_tokens(t)) for t in df["attributes"].astype(str)]

        jac_tt = np.zeros(n); ovl_tt = np.zeros(n); common_tt = np.zeros(n); cover_tt = np.zeros(n)
        jac_tc = np.zeros(n); cover_tcat = np.zeros(n); cover_tattr = np.zeros(n)
        term_in_title_substr = np.zeros(n)
        norm_terms = prep["term"]; norm_titles = prep["title"]
        for i in range(n):
            tt, ti, ca, at = term_tok[i], title_tok[i], cat_tok[i], attr_tok[i]
            jac_tt[i] = _jaccard(tt, ti)
            ovl_tt[i] = _overlap_coef(tt, ti)
            common_tt[i] = len(tt & ti)
            cover_tt[i] = (len(tt & ti) / len(tt)) if tt else 0.0
            jac_tc[i] = _jaccard(tt, ca)
            cover_tcat[i] = (len(tt & ca) / len(tt)) if tt else 0.0
            cover_tattr[i] = (len(tt & at) / len(tt)) if tt else 0.0
            term_in_title_substr[i] = 1.0 if norm_terms[i] and norm_terms[i] in norm_titles[i] else 0.0

        feats["jaccard_term_title"] = jac_tt
        feats["overlap_coef_term_title"] = ovl_tt
        feats["common_tokens_term_title"] = common_tt
        feats["coverage_term_in_title"] = cover_tt
        feats["jaccard_term_category"] = jac_tc
        feats["coverage_term_in_category"] = cover_tcat
        feats["coverage_term_in_attrs"] = cover_tattr
        feats["term_substring_of_title"] = term_in_title_substr

        # --- Uzunluk / sayım öznitelikleri ---
        term_len = np.array([len(t) for t in prep["term"]], dtype=float)
        title_len = np.array([len(t) for t in prep["title"]], dtype=float)
        feats["term_char_len"] = term_len
        feats["title_char_len"] = title_len
        feats["term_word_count"] = np.array([len(t.split()) for t in prep["term"]], dtype=float)
        feats["title_word_count"] = np.array([len(t.split()) for t in prep["title"]], dtype=float)
        feats["len_ratio_term_title"] = np.where(title_len > 0, term_len / (title_len + 1e-6), 0.0)

        # --- (Opsiyonel) embedding kosinüs benzerliği ---
        if self.use_embeddings and self._embedder is not None:
            emb_term = self._embedder.encode(prep["term"], normalize_embeddings=True,
                                             show_progress_bar=False, batch_size=256)
            emb_doc = self._embedder.encode(prep["product_doc"], normalize_embeddings=True,
                                            show_progress_bar=False, batch_size=256)
            feats["embed_cos_term_doc"] = (emb_term * emb_doc).sum(axis=1)

        out = pd.DataFrame(feats)
        self.feature_names_ = list(out.columns)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # --- kalıcılık ---
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "FeatureBuilder":
        with open(path, "rb") as f:
            return pickle.load(f)
