"""Retrieval tabanlı negatif madenciliği — eğitim setini TESTİN kurulduğu gibi kurar.

Test çiftleri bir arama/retrieval sisteminin çıktısıdır: her terim için ~104
"makul" aday, bunların ~%25-28'i insanlarca alakalı işaretlenmiş. Sentetik
negatif örnekleme (rastgele/kategori/kelime-örtüşme) bu dağılımı taklit edemez;
model sentetik negatifleri ayırmayı öğrenir, gerçek test negatiflerinde şaşırır
(v3'te OOF yükselirken LB'nin düşmesinin nedeni buydu).

Çözüm: her train terimi için katalogdan embedding benzerliğine göre en yakın
top-K ürünü çek; bilinen pozitifler 1, kalan adaylar 0. Böylece:
  * train grupları test gruplarıyla yapısal olarak aynı olur,
  * grup öznitelikleri ve karar eşiği doğal kalibre olur,
  * OOF skoru leaderboard'un güvenilir bir proxy'sine dönüşür.

Gürültü notu: retrieval'da en tepede çıkan ama etiketlenmemiş ürünler çoğu kez
aslında alakalıdır (etiketleme örneklemesi eksiksiz değildir). Bu yüzden pozitif
sonrası ilk `skip_top` aday negatif olarak KULLANILMAZ.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd


def mine_retrieval_negatives(
    pos: pd.DataFrame,
    items: pd.DataFrame,
    emb_terms_path,
    emb_items_path,
    top_k: int = 120,
    neg_per_pos: float = 2.0,
    skip_top: int = 3,
    term_chunk: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Her train terimi için embedding-retrieval adaylarından negatif çiftler üretir.

    pos: term, term_id, item_id kolonlu pozitif çiftler (metin kolonları dahil).
    items: loader.load_items çıktısı (item_id + metin/brand/gender/age kolonları).
    Dönüş: pos ile aynı şemada, label=0 satırlar.
    """
    rng = random.Random(seed)

    t_npz = np.load(emb_terms_path, allow_pickle=False)
    i_npz = np.load(emb_items_path, allow_pickle=False)
    term_ids_all = t_npz["ids"].tolist()
    term_vecs_all = t_npz["vecs"]
    item_ids = i_npz["ids"].tolist()
    item_vecs = i_npz["vecs"]  # (N_items, d) L2-normalize

    # Yalnız train'de geçen terimler
    term_pos_items: dict[str, set[str]] = {}
    term_text: dict[str, str] = {}
    for tid, term, iid in zip(pos["term_id"], pos["term"], pos["item_id"]):
        term_pos_items.setdefault(tid, set()).add(iid)
        term_text[tid] = term
    t_index = {tid: k for k, tid in enumerate(term_ids_all)}
    tids = [t for t in term_pos_items if t in t_index]
    T = term_vecs_all[[t_index[t] for t in tids]]  # (n_terms, d)

    # item_id -> items tablosu satırı (metadata için)
    meta = items.set_index("item_id")
    meta_cols = ["product_title", "category", "attributes", "brand", "gender", "age_group"]
    for c in meta_cols:
        if c not in meta.columns:
            meta[c] = ""

    neg_rows: list[dict] = []
    n_terms = len(tids)
    for lo in range(0, n_terms, term_chunk):
        hi = min(lo + term_chunk, n_terms)
        scores = T[lo:hi] @ item_vecs.T  # (chunk, N_items) kosinüs
        # top_k + pozitif sayısı kadar aday al (pozitifler elendikten sonra top_k kalsın)
        k_take = top_k + 40
        top_idx = np.argpartition(scores, -k_take, axis=1)[:, -k_take:]
        for r, ti in enumerate(range(lo, hi)):
            tid = tids[ti]
            banned = term_pos_items[tid]
            row_scores = scores[r, top_idx[r]]
            order = top_idx[r][np.argsort(-row_scores)]
            cands = [item_ids[j] for j in order if item_ids[j] not in banned]
            cands = cands[skip_top:top_k]  # tepe gürültüsünü at, top_k ile sınırla
            n_neg = max(1, round(neg_per_pos * len(banned)))
            if len(cands) > n_neg:
                cands = rng.sample(cands, n_neg)
            term = term_text[tid]
            for iid in cands:
                m = meta.loc[iid]
                neg_rows.append({
                    "term": term, "term_id": tid, "item_id": iid,
                    "product_title": m["product_title"], "category": m["category"],
                    "attributes": m["attributes"], "brand": m["brand"],
                    "gender": m["gender"], "age_group": m["age_group"],
                    "label": 0,
                })
        done = min(hi, n_terms)
        print(f"  [retrieval] {done}/{n_terms} terim", flush=True)

    out = pd.DataFrame(neg_rows)
    print(f"[retrieval] {len(out)} negatif ({n_terms} terim, top_k={top_k}, "
          f"skip_top={skip_top}, hedef oran={neg_per_pos}/pozitif)")
    return out
