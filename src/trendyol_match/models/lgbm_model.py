"""LightGBM ikili sınıflandırma modeli.

Neden LightGBM?
  * Hız: yarışmanın 'Hız' ekseni için hafif ve hızlı çıkarım.
  * Açıklanabilirlik: 'Açıklanabilirlik' ekseni için öznitelik önemleri + SHAP.
  * Güçlü tablosal başarı: mühendisliği yapılmış özniteliklerde çok etkili.

Doğrulama: terim-bazlı GroupKFold ile sızıntı önlenir (aynı terim hem train hem
validasyonda olmaz), böylece skor test dağılımına daha sadık olur.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import GroupKFold, StratifiedKFold


class LgbmMatcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.params = dict(cfg.model.lgbm_params)
        self.n_folds = int(cfg.model.cv_folds)
        self.early_stopping_rounds = int(cfg.model.early_stopping_rounds)
        self.seed = int(cfg.model.seed)
        self.models_: list[LGBMClassifier] = []
        self.feature_names_: list[str] = []
        self.oof_: np.ndarray | None = None
        self.feature_importance_: pd.DataFrame | None = None

    def _make_folds(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None):
        if groups is not None and len(np.unique(groups)) >= self.n_folds:
            gkf = GroupKFold(n_splits=self.n_folds)
            return list(gkf.split(X, y, groups))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        return list(skf.split(X, y))

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None,
            verbose: bool = True) -> "LgbmMatcher":
        y = np.asarray(y)
        self.feature_names_ = list(X.columns)
        self.oof_ = np.zeros(len(X))
        importances = np.zeros(X.shape[1])
        folds = self._make_folds(X, y, groups)

        for fold, (tr_idx, va_idx) in enumerate(folds, 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model = LGBMClassifier(**self.params, random_state=self.seed)
            callbacks = [early_stopping(self.early_stopping_rounds, verbose=False)]
            if verbose:
                callbacks.append(log_evaluation(period=0))
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
            self.oof_[va_idx] = model.predict_proba(X_va)[:, 1]
            importances += model.feature_importances_
            self.models_.append(model)
            if verbose:
                from ..evaluation.metrics import evaluate, format_metrics
                m = evaluate(y_va, self.oof_[va_idx], threshold=0.5)
                print(f"  [fold {fold}/{len(folds)}] {format_metrics(m)}  "
                      f"(best_iter={model.best_iteration_})")

        self.feature_importance_ = (
            pd.DataFrame({"feature": self.feature_names_, "importance": importances / len(folds)})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Tüm fold modellerinin ortalaması (bagging etkisi)."""
        X = X[self.feature_names_]
        preds = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return preds

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "LgbmMatcher":
        with open(path, "rb") as f:
            return pickle.load(f)
