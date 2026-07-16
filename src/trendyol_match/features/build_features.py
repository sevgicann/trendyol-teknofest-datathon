"""Öznitelik mühendisliği.

Terim–ürün çiftinden yorumlanabilir sayısal öznitelikler üretir:
  * Sözlüksel örtüşme (Jaccard, overlap katsayısı, ortak token, kapsama)
  * TF-IDF kosinüs benzerliği (kelime n-gram + karakter n-gram)
  * Yapılandırılmış eşleşme: marka / cinsiyet / yaş uyumu ve ÇATIŞMASI
    (terimlerin ~%63'ü marka içerir; "kadın mont" sorgusuna erkek mont
    göstermek klasik precision hatasıdır — bu öznitelikler onu hedefler)
  * Uzunluk / sayım öznitelikleri
  * (Opsiyonel) önceden hesaplanmış embedding kosinüsü (term_id/item_id ile)
  * Terim-içi GRUP öznitelikleri (add_group_features): testte her terim ~104
    aday ürünle gelir; alaka görecelidir — adayın benzerlik skoru terimin
    diğer adaylarına göre sıralanır (rank-percentile, ortalama/maks. fark)

FeatureBuilder durumludur (stateful): TF-IDF sözlüğü ve marka sözlüğü train'de
FIT edilir, test'te aynı yapılarla TRANSFORM edilir. Açıklanabilirlik için tüm
öznitelik adları anlamlıdır.
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


def _col_or_empty(df: pd.DataFrame, name: str) -> list[str]:
    if name in df.columns:
        return df[name].fillna("").astype(str).tolist()
    return [""] * len(df)


# Terim metninden cinsiyet/yaş niyeti çıkarımı (deaccent edilmiş token'larla)
_FEMALE_TOKENS = {"kadin", "bayan", "kiz"}
_MALE_TOKENS = {"erkek", "bay", "oglan"}
_AGE_TOKENS = {"bebek": "bebek", "cocuk": "cocuk", "genc": "genc"}


def _term_gender(tokens: set[str]) -> str:
    f = bool(tokens & _FEMALE_TOKENS)
    m = bool(tokens & _MALE_TOKENS)
    if f and not m:
        return "kadin"
    if m and not f:
        return "erkek"
    return ""


def _term_age(tokens: set[str]) -> str:
    for tok, age in _AGE_TOKENS.items():
        if tok in tokens:
            return age
    return ""


class FeatureBuilder:
    # Grup özniteliklerinin türetileceği taban kolonlar
    GROUP_BASE_COLS = ["tfidf_char_cos_term_doc", "tfidf_word_cos_term_doc"]

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
        self.brand_vocab_: frozenset[str] = frozenset()
        self.emb_terms_path = fcfg.get("emb_terms_file")
        self.emb_items_path = fcfg.get("emb_items_file")
        self._emb_terms: dict[str, np.ndarray] | None = None
        self._emb_items: dict[str, np.ndarray] | None = None
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
        # Marka sözlüğü: "terim BAŞKA bir marka mı istiyor?" özniteliği için
        brands = {normalize(b, deaccent=True) for b in _col_or_empty(df, "brand")}
        self.brand_vocab_ = frozenset(b for b in brands
                                      if len(b) >= 3 and b != "unknown")
        return self

    # --- embedding sözlükleri (önceden hesaplanmış, ID ile aranır) ---
    def _load_embeddings(self) -> None:
        if self._emb_terms is not None or not self.emb_terms_path:
            return
        for attr, rel in (("_emb_terms", self.emb_terms_path),
                          ("_emb_items", self.emb_items_path)):
            path = Path(self.cfg.get("_root", ".")) / rel
            if path.exists():
                data = np.load(path, allow_pickle=False)
                ids = data["ids"]
                vecs = data["vecs"]
                setattr(self, attr, dict(zip(ids.tolist(), vecs)))
                print(f"[embed] yüklendi: {path.name} ({len(ids)} vektör)")
            else:
                setattr(self, attr, {})

    def _embedding_cos(self, df: pd.DataFrame) -> np.ndarray | None:
        """term_id/item_id varsa önceden hesaplanmış embedding kosinüsü."""
        if not self.emb_terms_path or "term_id" not in df.columns or "item_id" not in df.columns:
            return None
        self._load_embeddings()
        if not self._emb_terms or not self._emb_items:
            return None
        out = np.full(len(df), np.nan)
        t_ids = df["term_id"].tolist()
        i_ids = df["item_id"].tolist()
        for k in range(len(df)):
            tv = self._emb_terms.get(t_ids[k])
            iv = self._emb_items.get(i_ids[k])
            if tv is not None and iv is not None:
                out[k] = float(np.dot(tv, iv))
        return out

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

        # --- IDF-ağırlıklı kapsama: nadir kelime eşleşmesi ("nike") yaygın
        # kelime eşleşmesinden ("siyah") çok daha güçlü alaka kanıtıdır ---
        vocab = self.word_vec.vocabulary_
        idf = self.word_vec.idf_
        max_idf = float(idf.max()) if len(idf) else 1.0

        def _idf(tok: str) -> float:
            j = vocab.get(tok)
            return float(idf[j]) if j is not None else max_idf  # OOV = nadir

        idf_cover_title = np.zeros(n)
        idf_cover_doc = np.zeros(n)
        for i in range(n):
            tt = term_tok[i]
            if not tt:
                continue
            total = sum(_idf(t) for t in tt)
            if total <= 0:
                continue
            in_title = sum(_idf(t) for t in tt if t in title_tok[i])
            doc_set = title_tok[i] | cat_tok[i] | attr_tok[i]
            in_doc = sum(_idf(t) for t in tt if t in doc_set)
            idf_cover_title[i] = in_title / total
            idf_cover_doc[i] = in_doc / total
        feats["idf_coverage_term_in_title"] = idf_cover_title
        feats["idf_coverage_term_in_doc"] = idf_cover_doc

        # --- Yapılandırılmış eşleşme: marka / cinsiyet / yaş ---
        brands = [normalize(b, deaccent=True) for b in _col_or_empty(df, "brand")]
        genders = [normalize(g, deaccent=True) for g in _col_or_empty(df, "gender")]
        ages = [normalize(a, deaccent=True) for a in _col_or_empty(df, "age_group")]
        terms_da = [normalize(t, deaccent=True) for t in df["term"].astype(str)]

        brand_in_term = np.zeros(n)
        term_wants_other_brand = np.zeros(n)
        gender_match = np.zeros(n); gender_conflict = np.zeros(n)
        age_match = np.zeros(n); age_conflict = np.zeros(n)

        # terim başına önbellek (terimler çok tekrarlanır)
        term_cache: dict[str, tuple[bool, str, str, set[str]]] = {}
        vocab = self.brand_vocab_
        for i in range(n):
            t = terms_da[i]
            cached = term_cache.get(t)
            if cached is None:
                toks = t.split()
                tokset = set(toks)
                # terimdeki 1-3 kelimelik n-gram'lardan marka sözlüğünde olanlar
                mentioned: set[str] = set()
                for a in range(len(toks)):
                    for b in (a + 1, a + 2, a + 3):
                        if b > len(toks):
                            break
                        ng = " ".join(toks[a:b])
                        if len(ng) >= 3 and ng in vocab:
                            mentioned.add(ng)
                cached = (bool(mentioned), _term_gender(tokset), _term_age(tokset), mentioned)
                term_cache[t] = cached
            has_brand_mention, tg, ta, mentioned = cached

            b = brands[i]
            if b and b != "unknown" and len(b) >= 3 and f" {b} " in f" {t} ":
                brand_in_term[i] = 1.0
            elif has_brand_mention:
                term_wants_other_brand[i] = 1.0

            g = genders[i]
            if tg:
                if g == tg or g == "unisex":
                    gender_match[i] = 1.0
                elif g in ("kadin", "erkek"):
                    gender_conflict[i] = 1.0

            a_ = ages[i]
            if ta:
                if ta in a_:
                    age_match[i] = 1.0
                elif a_ == "yetiskin":
                    age_conflict[i] = 1.0

        feats["brand_in_term"] = brand_in_term
        feats["term_wants_other_brand"] = term_wants_other_brand
        feats["gender_match"] = gender_match
        feats["gender_conflict"] = gender_conflict
        feats["age_match"] = age_match
        feats["age_conflict"] = age_conflict

        # --- Uzunluk / sayım öznitelikleri ---
        term_len = np.array([len(t) for t in prep["term"]], dtype=float)
        title_len = np.array([len(t) for t in prep["title"]], dtype=float)
        feats["term_char_len"] = term_len
        feats["title_char_len"] = title_len
        feats["term_word_count"] = np.array([len(t.split()) for t in prep["term"]], dtype=float)
        feats["title_word_count"] = np.array([len(t.split()) for t in prep["title"]], dtype=float)
        feats["len_ratio_term_title"] = np.where(title_len > 0, term_len / (title_len + 1e-6), 0.0)

        # --- (Opsiyonel) önceden hesaplanmış embedding kosinüsü ---
        emb_cos = self._embedding_cos(df)
        if emb_cos is not None:
            feats["embed_cos_term_item"] = emb_cos

        out = pd.DataFrame(feats)
        self.feature_names_ = list(out.columns)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # --- Terim-içi grup öznitelikleri --------------------------------------
    @staticmethod
    def add_group_features(X: pd.DataFrame, terms: pd.Series | list) -> pd.DataFrame:
        """Adayın benzerlik skorlarını AYNI TERİMİN diğer adaylarına göre konumlar.

        Test her terim için ~104 aday içerir; 'bu aday, terimin adayları içinde
        ne kadar iyi?' sorusu mutlak skordan daha ayırt edicidir. Rank-percentile
        ölçekten bağımsızdır; ortalama/maksimuma uzaklık marjı ölçer.
        Train'de negatif oranı test dengesine yakın seçilirse dağılım uyumu korunur.
        """
        g = pd.Series(list(terms), index=X.index)
        for col in FeatureBuilder.GROUP_BASE_COLS:
            if col not in X.columns:
                continue
            grp = X[col].groupby(g)
            X[f"grp_rank_{col}"] = grp.rank(pct=True).values
            mean = grp.transform("mean")
            mx = grp.transform("max")
            X[f"grp_delta_mean_{col}"] = (X[col] - mean).values
            X[f"grp_delta_max_{col}"] = (X[col] - mx).values
        return X

    # --- kalıcılık ---
    def save(self, path: str | Path) -> None:
        # Embedding sözlükleri pickle'a girmez (büyük); yolları kalır, tekrar yüklenir
        emb_t, emb_i = self._emb_terms, self._emb_items
        self._emb_terms = self._emb_items = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        finally:
            self._emb_terms, self._emb_items = emb_t, emb_i

    @staticmethod
    def load(path: str | Path) -> "FeatureBuilder":
        with open(path, "rb") as f:
            return pickle.load(f)
