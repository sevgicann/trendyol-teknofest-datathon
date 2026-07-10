# Trendyol TEKNOFEST E-Ticaret Yarışması 2026 — Datathon (Kaggle)

**Ürün–Terim Alaka Tahmini / Product–Term Relevance Prediction**

Bir arama terimi (search term) ile bir ürün (product) arasındaki **anlamsal ilişkiyi** ikili
sınıflandırma ile tahmin eden uçtan uca (end-to-end) makine öğrenmesi projesi.

> **Görev:** Verilen bir `(terim, ürün)` çifti için **alakalı (1)** mı yoksa **alakasız (0)** mı
> olduğunu tahmin et.
>
> **Kritik nokta:** Eğitim verisi **yalnızca alakalı (pozitif) çiftleri** içerir. Test verisi hem
> alakalı hem alakasız çiftleri içerir. Bu yüzden modelleme **negatif örnekleme (negative sampling)**
> üzerine kuruludur.

## Değerlendirme kriterleri
Yarışma üç eksende puanlanır — mimarimiz üçünü birden gözetir:
1. **Başarı (Success):** Sınıflandırma başarısı (F1 / accuracy).
2. **Hız (Speed):** Hafif, hızlı çıkarım → LightGBM + vektörel öznitelikler.
3. **Açıklanabilirlik (Explainability):** Yorumlanabilir öznitelikler + LightGBM önem skorları / SHAP.

## Gerçek veri (ilişkisel şema)
Yarışma verisi `folder/` altında, **ilişkisel** yapıdadır — çift dosyaları yalnız ID taşır,
metinler sözlük tablolarından join edilir (loader bunu otomatik yapar):

| Dosya | İçerik | Boyut |
|---|---|---|
| `folder/terms.csv` | `term_id, query` — arama terimi metinleri | 50.153 terim |
| `folder/items.csv` | `item_id, title, category, brand, gender, age_group, attributes` | 962.873 ürün |
| `folder/training_pairs.csv` | `id, term_id, item_id, label` — **yalnız pozitif** (label=1) | 250.000 çift |
| `folder/submission_pairs.csv` | `id, term_id, item_id` — tahmin edilecek çiftler | 3.359.679 çift |
| `folder/sample_submission.csv` | `id, prediction` — beklenen çıktı formatı | 3.359.679 satır |

`brand / gender / age_group` alanları `attributes` metnine katılır; böylece marka
eşleşmesi gibi sinyaller mevcut öznitelikler (TF-IDF, kapsama) üzerinden modele akar.
Negatif örnekleme, havuz olarak **tam ürün kataloğunu** kullanır (test kataloğun
~%97'sini kapsar). 3,36M test çifti bellek dostu parçalar halinde puanlanır
(`config.yaml → inference.chunk_size`).

## Proje yapısı
```
teknofest/
├── config/config.yaml          # Kolon eşlemesi + tüm hiperparametreler (şemaya uyarlanabilir)
├── folder/                      # GERÇEK yarışma verisi (ilişkisel: terms/items/pairs)
├── data/
│   ├── raw/                     # Sentetik/deneme verisi (pipeline provası için)
│   ├── interim/                 # Negatif örneklenmiş / temizlenmiş veri
│   └── processed/               # Öznitelik matrisleri
├── models/                      # Eğitilmiş modeller + artefaktlar
├── submissions/                 # Kaggle submission dosyaları
├── reports/figures/             # EDA grafikleri
├── src/trendyol_match/          # Ana Python paketi
│   ├── config.py                # config.yaml yükleyici
│   ├── data/                    # download, synthetic, loader, negative_sampling
│   ├── features/                # text_normalize, build_features
│   ├── models/                  # lgbm_model
│   ├── evaluation/              # metrics
│   └── pipeline/                # eda, train, predict
└── scripts/                     # 00_download → 01_synthetic → 02_eda → 03_train → 04_predict
```

## Kurulum
```bash
python -m pip install -r requirements.txt
```

## Çalıştırma (sırasıyla)
```bash
# 0) Gerçek veri folder/ altında hazır. (Alternatif: sentetik veri üretip
#    config.yaml'daki yolları data/raw'a çevirerek pipeline provası yapılabilir.)

# 1) Keşifsel veri analizi
python scripts/02_eda.py

# 2) Modeli eğit (negatif örnekleme + öznitelik + LightGBM CV + eşik ayarı)
python scripts/03_train.py

# 3) Test için tahmin üret → submissions/submission.csv
python scripts/04_predict.py
```

## Kaggle verisini indirme
1. Kaggle hesabınızdan **Account → Create New API Token** ile `kaggle.json` indirin.
2. Dosyayı `C:\Users\<kullanıcı>\.kaggle\kaggle.json` konumuna koyun.
3. Yarışma kurallarını Kaggle sitesinden kabul edin.
4. `python scripts/00_download_data.py` çalıştırın (veya `kaggle competitions download -c trendyol-e-ticaret-yarismasi-2026-kaggle`).

Gerçek verinin kolon adları resmi olandan farklıysa **kod değişmez** — sadece
`config/config.yaml` içindeki `columns:` eşlemesini güncelleyin.

## Yol haritası
Ayrıntılı, madde madde plan için bkz. [PROJECT_CHECKLIST.md](PROJECT_CHECKLIST.md).
