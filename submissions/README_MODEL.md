# En İyi Model Paketi — Üçlü Harman (LB F1 = 0.767)

**Trendyol TEKNOFEST E-Ticaret Yarışması 2026 (Kaggle) — Takım: TeamX**

## En iyi submission

| Dosya | Leaderboard F1 | Açıklama |
|---|---|---|
| `submission_blend3_pos31.csv` | **0.767** | v2+v5+v6 rank-ortalama harmanı, pozitif oran %31 |
| `submission_blend3_pos33.csv` | 0.762 | Aynı harman, pozitif oran %33 |

## Nasıl üretildi?

Üç LightGBM modelinin test olasılıklarının **rank-ortalaması** alındı, hedef
pozitif oranına karşılık gelen kantil eşiğiyle 0/1'e çevrildi:

- **v2** (`test_probabilities_v2.csv`): sözlüksel öznitelikler (TF-IDF kelime +
  karakter kosinüs, Jaccard, kapsama), temiz sentetik negatifler
  (%40 terim-zor, %30 kategori-zor, %30 rastgele; negatif asla terimin pozitif
  kategorisinden gelmez), 1:1 denge. Tek başına LB: 0.750 (@%33).
- **v5** (`test_probabilities_v5.csv`): v2 + embedding kosinüsü
  (paraphrase-multilingual-MiniLM-L12-v2; 50K terim + 963K ürün bir kez encode)
  + marka/cinsiyet/yaş eşleşme-çatışma öznitelikleri.
- **v6** (`test_probabilities_v6.csv`): v5 + pseudo-labeling (v2 ve v5'in
  mutabık+emin olduğu 400K gerçek test çifti eğitime katıldı) + IDF-ağırlıklı
  kapsama öznitelikleri.

Harman olasılıkları: `test_probabilities_blend3.csv`
(başka eşikte submission üretmek için yeterlidir).

## Paketteki model artefaktları (v6)

- `lgbm_matcher.pkl` — 5-fold LightGBM topluluğu (GroupKFold, terim bazlı)
- `feature_builder.pkl` — TF-IDF sözlükleri + marka sözlüğü + öznitelik üretici
- `train_meta.json` — eşik ve OOF metrikleri
- `feature_importance.csv` — açıklanabilirlik: öznitelik önemleri

## Yeniden üretim

Kod: https://github.com/sevgicann/trendyol-teknofest-datathon (dal: v6-pseudo-labeling)

```bash
python scripts/06_embed.py    # embedding'ler (bir kez, ~2-4 saat CPU)
python scripts/03_train.py    # eğitim (pseudo-labeling config'de açık)
python scripts/04_predict.py  # test olasılıkları + submission
python scripts/05_threshold_sweep.py 0.31  # istenen pozitif oranda eşik
```

Not: embedding .npz dosyaları (1.3GB) pakete dahil değildir; `06_embed.py`
ile yeniden üretilir.
