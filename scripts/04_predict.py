"""Test verisini puanlar ve submissions/submission.csv üretir."""
import _bootstrap  # noqa: F401
from trendyol_match.pipeline.predict import run_predict

if __name__ == "__main__":
    run_predict()
