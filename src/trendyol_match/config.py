"""Konfigürasyon yükleyici.

config/config.yaml dosyasını okur ve nokta-erişimli (dot-access) bir sözlük döndürür.
Proje kökünü otomatik bulur, böylece scriptler herhangi bir dizinden çalıştırılabilir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Bu dosyadan yukarı çıkarak proje kökünü bulur (config/ klasörünü arar)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Yedek: paket iki seviye altta (src/trendyol_match/config.py)
    return here.parents[2]


class Config(dict):
    """Nokta erişimli sözlük: cfg.paths.train_file gibi kullanılır."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str | Path | None = None) -> Config:
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw)
    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: Config, relative: str) -> Path:
    """config içindeki göreli yolu proje köküne göre mutlak yola çevirir."""
    return Path(cfg["_root"]) / relative
