"""Türkçe metin normalizasyonu.

E-ticaret arama metinleri için: küçük harfe indirgeme (Türkçe i/İ kuralı),
noktalama temizliği, boşluk normalizasyonu ve opsiyonel aksan sadeleştirme.
"""
from __future__ import annotations

import re
import unicodedata

# Türkçe'ye özgü küçük harf eşlemesi (I->ı, İ->i) — str.lower() bunu yanlış yapar
_TR_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_ASCII_MAP = str.maketrans("ışğüöçâîû", "isguocaiu")


def turkish_lower(text: str) -> str:
    return text.translate(_TR_LOWER_MAP).lower()


def normalize(text: str, deaccent: bool = False) -> str:
    """Metni normalize eder.

    deaccent=True ise Türkçe karakterleri ASCII karşılığına indirger
    (ör. 'ayakkabı' -> 'ayakkabi'); terim/başlık eşleşmesinde yazım
    varyasyonlarını yakalamak için faydalı olabilir.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = turkish_lower(text)
    text = _PUNCT_RE.sub(" ", text)
    if deaccent:
        text = text.translate(_ASCII_MAP)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, deaccent: bool = False) -> list[str]:
    norm = normalize(text, deaccent=deaccent)
    return norm.split() if norm else []


# Çok yaygın ve ayırt edici olmayan Türkçe sözcükler (stopword — hafif liste)
STOPWORDS = {
    "ve", "ile", "için", "bir", "bu", "da", "de", "the", "of", "cm", "adet",
}


def content_tokens(text: str, deaccent: bool = False) -> list[str]:
    """Stopword'ler çıkarılmış, uzunluğu 1'den büyük anlamlı token'lar."""
    return [t for t in tokenize(text, deaccent=deaccent) if t not in STOPWORDS and len(t) > 1]
