"""Sentetik (Trendyol-benzeri) veri üretir → data/raw/ altına yazar.

Gerçek Kaggle verisi inmeden pipeline'ı uçtan uca test etmek için kullanılır.
Gerçek veri geldiğinde bu adımı atlayın; loader gerçek dosyaları okuyacaktır.
"""
import _bootstrap  # noqa: F401
from trendyol_match.data.make_synthetic import write_synthetic

if __name__ == "__main__":
    write_synthetic()
