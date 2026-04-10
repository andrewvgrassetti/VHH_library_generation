import json
import csv
import os
from pathlib import Path
from datetime import datetime
import pandas as pd


class LibraryManager:
    def __init__(self, session_id: str = None):
        if session_id is None:
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self._session_id = session_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def create_variant_id(self, variant_number: int) -> str:
        return f"VHH-{self._session_id}-{variant_number:06d}"

    def save_session(self, data: dict, output_dir: str = "sessions") -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        base = Path(output_dir) / self._session_id
        paths = {}

        json_data = {}
        for k, v in data.items():
            if isinstance(v, pd.DataFrame):
                json_data[k] = v.to_dict(orient="records")
            else:
                json_data[k] = v
        json_path = str(base) + ".json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        paths["json"] = json_path

        variants_df = None
        for k, v in data.items():
            if isinstance(v, pd.DataFrame) and "aa_sequence" in v.columns:
                variants_df = v
                break
        if variants_df is not None:
            csv_path = str(base) + ".csv"
            self.export_csv(variants_df, csv_path)
            paths["csv"] = csv_path

        if variants_df is not None and "aa_sequence" in variants_df.columns:
            fasta_path = str(base) + ".fasta"
            self.export_fasta(variants_df, fasta_path)
            paths["fasta"] = fasta_path

        return paths

    def load_session(self, filepath: str) -> dict:
        with open(filepath) as f:
            return json.load(f)

    def export_csv(self, variants: pd.DataFrame, filepath: str):
        variants.to_csv(filepath, index=False)

    def export_fasta(self, variants: pd.DataFrame, filepath: str, sequence_col: str = "aa_sequence"):
        with open(filepath, "w") as f:
            for idx, row in variants.iterrows():
                header = row["variant_id"] if "variant_id" in row.index else f"variant_{idx}"
                seq = row[sequence_col]
                f.write(f">{header}\n{seq}\n")
