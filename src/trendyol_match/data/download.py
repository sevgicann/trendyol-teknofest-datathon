"""Kaggle yarışma verisini indirir.

Gereksinim: `pip install kaggle` ve `~/.kaggle/kaggle.json` (API token).
Windows'ta konum: C:\\Users\\<kullanici>\\.kaggle\\kaggle.json
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from ..config import load_config, resolve_path


def kaggle_credentials_present() -> bool:
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    import os

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return any(p.exists() for p in candidates)


def download_competition_data(cfg=None) -> Path:
    cfg = cfg or load_config()
    slug = cfg.competition.slug
    raw_dir = resolve_path(cfg, cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not kaggle_credentials_present():
        print(
            "\n[UYARI] Kaggle kimlik bilgisi bulunamadı.\n"
            "  1) Kaggle > Account > 'Create New API Token' ile kaggle.json indirin\n"
            f"  2) {Path.home() / '.kaggle' / 'kaggle.json'} konumuna koyun\n"
            "  3) Yarışma kurallarını Kaggle sitesinden kabul edin\n"
            "  Alternatif: verileri elle indirip data/raw/ altına çıkarın, ya da\n"
            "  'python scripts/01_make_synthetic.py' ile sentetik veriyle çalışın.\n"
        )
        return raw_dir

    print(f"[indir] Kaggle yarışması: {slug}")
    cmd = [
        sys.executable, "-m", "kaggle", "competitions", "download",
        "-c", slug, "-p", str(raw_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("[hata] İndirme başarısız. Kurallar kabul edildi mi? Slug doğru mu?")
        return raw_dir

    # İnen zip'leri aç
    for zip_path in raw_dir.glob("*.zip"):
        print(f"[çıkar] {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    print(f"[tamam] Veri hazır: {raw_dir}")
    for f in sorted(raw_dir.iterdir()):
        if f.suffix != ".zip":
            print(f"   - {f.name}")
    return raw_dir


if __name__ == "__main__":
    download_competition_data()
