from pathlib import Path

import pandas as pd


class DatasetLoader:
    """Load headline JSONL; resolve paths from repo root. Prefer get_dataframe() for analysis."""

    def __init__(self, dataset_path: str | Path = "dataset/Sarcasm_Headlines_Dataset_v2.json"):
        self.dataset_path = (Path(__file__).resolve().parent.parent / Path(dataset_path)).resolve()
        self._df: pd.DataFrame | None = None

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_json(self.dataset_path, lines=True, encoding="utf-8")
        return self._df

    def get_dataframe(self) -> pd.DataFrame:
        return self._ensure_loaded()

    def load_data(self) -> list[dict]:
        """List of row dicts (handy for JSONL-style APIs). Prefer get_dataframe() when possible."""
        return self._ensure_loaded().to_dict(orient="records")

    def get_data(self) -> list[dict]:
        return self.load_data()

    def get_data_size(self) -> int:
        return len(self._ensure_loaded())

    def get_data_sample(self, index: int) -> dict:
        return self._ensure_loaded().iloc[index].to_dict()
