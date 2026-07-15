"""Embedding ön-hesaplama — benzersiz terim ve ürünleri BİR KEZ encode eder.

3.36M test çiftini satır satır encode etmek CPU'da günler sürer; oysa çiftler
yalnızca 50K benzersiz terim × 963K benzersiz üründen oluşur. Her metni bir kez
encode edip ID -> vektör sözlüğü olarak kaydederiz; FeatureBuilder çift başına
sadece bir iç çarpım (kosinüs) yapar.

Vektörler L2-normalize kaydedilir (dot = cosine). Ürün metni: başlık + kategori
(attributes gürültülü ve uzun — anlamsal imzayı başlık+kategori taşır).

Kullanım:  python scripts/06_embed.py
Çıktı:     models/emb_terms.npz, models/emb_items.npz  (ids + float32 vecs)
"""
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from trendyol_match.config import load_config, resolve_path


SHARD = 50_000  # kesintiye dayanıklılık: her ~50K metin ayrı parça olarak kaydedilir


def encode_and_save(model, ids: list[str], texts: list[str], out_path, batch_size=256) -> None:
    """Metinleri parça parça encode eder; mevcut parçaları atlar (resume)."""
    if out_path.exists():
        print(f"[embed] {out_path.name} zaten var, atlanıyor")
        return
    shard_dir = out_path.parent / (out_path.stem + "_shards")
    shard_dir.mkdir(exist_ok=True)
    n_shards = (len(texts) + SHARD - 1) // SHARD
    for s in range(n_shards):
        spath = shard_dir / f"shard_{s:03d}.npz"
        if spath.exists():
            print(f"[embed] parça {s+1}/{n_shards} mevcut, atlandı")
            continue
        t0 = time.time()
        lo, hi = s * SHARD, min((s + 1) * SHARD, len(texts))
        vecs = model.encode(texts[lo:hi], batch_size=batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
        np.savez(spath, ids=np.array(ids[lo:hi]), vecs=vecs.astype(np.float32))
        print(f"[embed] parça {s+1}/{n_shards} ({hi-lo} metin, {time.time()-t0:.0f}s)")
    # parçaları birleştir ve tek dosyaya yaz
    all_ids, all_vecs = [], []
    for s in range(n_shards):
        data = np.load(shard_dir / f"shard_{s:03d}.npz", allow_pickle=False)
        all_ids.append(data["ids"]); all_vecs.append(data["vecs"])
    np.savez_compressed(out_path, ids=np.concatenate(all_ids),
                        vecs=np.concatenate(all_vecs))
    print(f"[embed] {out_path.name}: {sum(len(a) for a in all_ids)} vektör birleştirildi")


def main() -> None:
    from sentence_transformers import SentenceTransformer

    cfg = load_config()
    model_name = cfg.features["embedding_model"]
    print(f"[embed] model yükleniyor: {model_name}")
    model = SentenceTransformer(model_name)

    terms = pd.read_csv(resolve_path(cfg, cfg.paths.terms_file), encoding="utf-8")
    t_id, t_text = terms.columns[0], terms.columns[1]
    encode_and_save(model, terms[t_id].tolist(), terms[t_text].astype(str).tolist(),
                    resolve_path(cfg, cfg.features["emb_terms_file"]))

    items = pd.read_csv(resolve_path(cfg, cfg.paths.items_file), encoding="utf-8",
                        usecols=["item_id", "title", "category"])
    docs = (items["title"].fillna("") + " " + items["category"].fillna("")).tolist()
    encode_and_save(model, items["item_id"].tolist(), docs,
                    resolve_path(cfg, cfg.features["emb_items_file"]))


if __name__ == "__main__":
    main()
