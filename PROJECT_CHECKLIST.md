# Proje Kontrol Listesi — Trendyol E-Ticaret 2026 Datathon

Senior data scientist yaklaşımıyla, her adım bir öncekinin üzerine kurulur.
Durum: `[x]` tamam · `[~]` devam ediyor · `[ ]` bekliyor

## Faz 0 — Anlama & Kurulum
- [x] Yarışmayı analiz et: görev, metrik, veri şeması, kısıtlar
- [x] Görev tespiti: **ürün–terim alaka**, ikili sınıflandırma, **pozitif-only train**
- [x] Ortam kontrolü: Python 3.14, pandas/sklearn/lightgbm, torch (CPU)
- [x] Proje iskeleti, config, requirements, README, checklist

## Faz 1 — Veri Erişimi
- [x] Kaggle indirme scripti + `kaggle.json` kurulum talimatı (`00_download_data.py`)
- [x] Sentetik veri üreteci — gerçek şemayı taklit eder, pipeline'ı hemen çalıştırılabilir kılar (`01_make_synthetic.py`)
- [x] Şemaya uyarlanabilir veri yükleyici — kolon adlarını otomatik tespit eder (`loader.py`)
- [x] **Gerçek veri entegrasyonu** — `folder/` altındaki ilişkisel şema (terms/items/pairs)
      loader'da otomatik join edilir; brand/gender/age_group attributes'a katılır

## Faz 2 — Keşifsel Veri Analizi (EDA)
- [x] Temel istatistikler: satır sayısı, benzersiz terim/ürün/kategori, eksik değer
- [x] Metin uzunlukları, kelime sayıları, kategori dağılımı
- [x] Terim–başlık sözcük örtüşmesi dağılımı (sinyal kontrolü)
- [x] EDA çıktıları `reports/figures/` ve konsol özeti (`02_eda.py`)

## Faz 3 — Ön İşleme & Etiket Kurgusu
- [x] Türkçe metin normalizasyonu (küçük harf, aksan/işaret, boşluk, i/İ) (`text_normalize.py`)
- [x] Negatif örnekleme: rastgele (kolay) + aynı-kategori (zor) negatifler (`negative_sampling.py`)
- [x] Sızıntısız doğrulama kurgusu: terim bazlı GroupKFold

## Faz 4 — Öznitelik Mühendisliği
- [x] Sözlüksel: token Jaccard, örtüşme oranı, ortak sözcük sayısı
- [x] TF-IDF (kelime + karakter n-gram) kosinüs benzerliği — Türkçe morfoloji için
- [x] Kategori/nitelik eşleşme öznitelikleri
- [x] Uzunluk / sayım öznitelikleri
- [x] (Opsiyonel) Çok dilli embedding kosinüs benzerliği (`build_features.py`)

## Faz 5 — Modelleme
- [x] LightGBM ikili sınıflandırıcı (hız + açıklanabilirlik dostu)
- [x] GroupKFold CV, early stopping, OOF tahminleri
- [x] F1'i maksimize eden karar eşiği seçimi (`lgbm_model.py`, `train.py`)
- [x] Öznitelik önemleri raporu (açıklanabilirlik)

## Faz 6 — Çıkarım & Submission
- [x] Test öznitelikleri (train ile birebir aynı boru hattı)
- [x] Tahmin + eşik uygulama → `submissions/submission.csv`
- [x] Submission formatını `sample_submission` ile doğrula (`predict.py`)

## Faz 7 — Doğrulama & İyileştirme
- [x] Tüm boru hattını sentetik veride uçtan uca çalıştır ve doğrula
- [x] Gerçek veriyi (`folder/`, 250K pozitif / 3.36M test) pipeline üzerinde uçtan uca çalıştır
      → OOF F1=0.9808 (eşik 0.4967, ROC-AUC 0.9969), 5 fold tutarlı
      → negatif örnekleme tam katalogdan (962K ürün), 250K pozitif için ~3 sn
      → submission üretildi ve formatı doğrulandı (3.359.679 satır, id sırası birebir)
- [x] Leaderboard iterasyonları (LB günlüğü):
      · v1 rastgele+kategori negatifleri → 0.671
      · v2 + terim-zor negatifler → 0.732; eşik taraması @%33-31 → 0.750
      · v3 grup öznitelikleri + ratio 3 → 0.714 (ders: sentetik grup yapısı testle uyuşmuyor)
      · v4 retrieval-madenli negatifler → 0.650 (ders: etiketlenmemiş-pozitif gürültüsü)
      · v5 = v2 tarifi + embedding + marka/cinsiyet → harman v2×v5 @%33 → 0.758
      · v6 + pseudo-labeling + IDF kapsama → üçlü harman @%31 → **0.767 (final)**
- [x] En iyi model paketi: `best_model_blend3_lb0767.zip` + `submissions/README_MODEL.md`
- [ ] (İkinci aşama — finale kalınırsa) Çok sınıflı sürüm: alakalı / az alakalı / alakasız
- [ ] (İkinci aşama) Cross-encoder / GPU ile embedding fine-tune

## Notlar
- Değerlendirme **Başarı + Hız + Açıklanabilirlik** — mimari üçünü de gözetir.
- Gerçek veri kolonları farrklıysa yalnızca `config/config.yaml → columns` güncellenir.
