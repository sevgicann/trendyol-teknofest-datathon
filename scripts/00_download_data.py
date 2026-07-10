"""Kaggle yarışma verisini indirir. Kimlik bilgisi yoksa yönerge basar."""
import _bootstrap  # noqa: F401
from trendyol_match.data.download import download_competition_data

if __name__ == "__main__":
    download_competition_data()
