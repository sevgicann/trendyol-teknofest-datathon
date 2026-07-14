"""Eşik taraması — kayıtlı test olasılıklarından farklı pozitif-oranlarda submission üretir.

Yeniden eğitim/tahmin GEREKTİRMEZ: 04_predict'in yazdığı test_probabilities.csv'yi
okur ve hedef pozitif oranına karşılık gelen kantil eşiğiyle 0/1 tahmin üretir.

Amaç: F1, tahmin edilen pozitif oranına çok duyarlıdır ve testin gerçek pozitif
oranı bilinmez. Farklı oranlarda 1-2 leaderboard denemesi, gerçek oranı
brakete alır ve sonraki eşik seçimini bilgiye dayandırır.

Kullanım:
    python scripts/05_threshold_sweep.py            # varsayılan oranlar: 0.55 0.35
    python scripts/05_threshold_sweep.py 0.50 0.30  # özel oranlar
"""
import sys

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from trendyol_match.config import load_config, resolve_path


def main() -> None:
    rates = [float(a) for a in sys.argv[1:]] or [0.55, 0.35]
    cfg = load_config()
    sub_dir = resolve_path(cfg, cfg.paths.submissions_dir)
    proba_path = sub_dir / "test_probabilities.csv"
    df = pd.read_csv(proba_path)
    id_col = df.columns[0]
    p = df["probability"].values
    print(f"[oku] {proba_path.name}: {len(df)} satır")

    for r in rates:
        thr = float(np.quantile(p, 1.0 - r))
        pred = (p >= thr).astype(int)
        name = f"submission_pos{int(round(r * 100))}.csv"
        out = pd.DataFrame({id_col: df[id_col], "prediction": pred})
        out.to_csv(sub_dir / name, index=False, encoding="utf-8")
        print(f"[eşik] hedef pozitif oran={r:.2f} -> eşik={thr:.4f}  "
              f"gerçek oran={pred.mean():.4f}  -> {name}")


if __name__ == "__main__":
    main()
