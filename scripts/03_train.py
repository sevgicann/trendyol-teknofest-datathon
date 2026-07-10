"""Modeli eğitir: negatif örnekleme → öznitelik → LightGBM CV → eşik seçimi."""
import _bootstrap  # noqa: F401
from trendyol_match.pipeline.train import run_train

if __name__ == "__main__":
    run_train()
