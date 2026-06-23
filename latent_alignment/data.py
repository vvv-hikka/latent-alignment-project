from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

DatasetFormat = Literal["auto", "paired", "polarity_raw"]


@dataclass(frozen=True)
class PairedDataset:
    positive_texts: list[str]
    negative_texts: list[str]
    labels: pd.Series
    pair_ids: pd.Series
    opposite_indices: pd.Series

    def __post_init__(self) -> None:
        n = len(self.positive_texts)
        if not (
            n
            == len(self.negative_texts)
            == len(self.labels)
            == len(self.pair_ids)
            == len(self.opposite_indices)
        ):
            raise ValueError("positive_texts, negative_texts, labels, and pair_ids must match")

    def train_test_indices(
        self,
        *,
        test_size: float = 0.15,
        random_state: int = 71,
        stratify: bool = True,
    ) -> tuple[pd.Index, pd.Index]:
        idx = pd.Index(range(len(self.labels)))
        stratify_labels = self.labels if stratify and self.labels.nunique() > 1 else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
        return pd.Index(train_idx), pd.Index(test_idx)


def load_dataset(
    path: str | Path,
    *,
    dataset_format: DatasetFormat = "auto",
    positive_col: str = "positive_text",
    negative_col: str = "negative_text",
    text_col: str = "statement",
    label_col: str = "is_harmfull_opposition",
    pair_id_col: str | None = None,
    positive_suffix: str = " Yes.",
    negative_suffix: str = " No.",
) -> PairedDataset:
    path = Path(path)
    df = _read_table(path)

    if dataset_format == "auto":
        if {positive_col, negative_col}.issubset(df.columns):
            dataset_format = "paired"
        elif text_col in df.columns:
            dataset_format = "polarity_raw"
        else:
            raise ValueError(
                "Could not infer dataset format. Use columns "
                f"{positive_col!r}/{negative_col!r}, or pass polarity raw data with {text_col!r}."
            )

    if dataset_format == "paired":
        return _load_paired(
            df,
            positive_col,
            negative_col,
            label_col,
            pair_id_col,
            positive_suffix,
            negative_suffix,
        )
    if dataset_format == "polarity_raw":
        return _load_polarity_raw(df, text_col, label_col, positive_suffix, negative_suffix)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def load_statements(
    path: str | Path,
    *,
    text_col: str = "statement",
    label_col: str = "is_harmfull_opposition",
) -> pd.DataFrame:
    """Load raw statements (without Yes/No suffixes) for free-form generation.

    Returns a DataFrame with columns ``statement``, ``label`` and ``pair_id``. Like the
    ``polarity_raw`` format, the file is treated as two stacked halves: the first half and
    the second half share ``pair_id`` values so opposite-polarity statements can be matched.
    Unlike :func:`load_dataset`, no suffix is appended and an even row count is not required.
    """
    path = Path(path)
    df = _read_table(path)
    _require_columns(df, [text_col])

    n = len(df)
    midpoint = n // 2
    if label_col in df:
        labels = df[label_col].astype(int).reset_index(drop=True)
    else:
        labels = pd.Series([0] * n, name=label_col)
    pair_ids = list(range(midpoint)) + list(range(n - midpoint))

    return pd.DataFrame(
        {
            "statement": df[text_col].astype(str).reset_index(drop=True),
            "label": labels,
            "pair_id": pd.Series(pair_ids),
        }
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def _load_paired(
    df: pd.DataFrame,
    positive_col: str,
    negative_col: str,
    label_col: str,
    pair_id_col: str | None,
    positive_suffix: str,
    negative_suffix: str,
) -> PairedDataset:
    _require_columns(df, [positive_col, negative_col])
    first_texts = df[positive_col].astype(str).tolist()
    second_texts = df[negative_col].astype(str).tolist()
    base_texts = first_texts + second_texts
    n_pairs = len(df)

    if label_col in df:
        first_labels = df[label_col].astype(int).tolist()
        second_labels = [1 - label for label in first_labels]
    else:
        first_labels = [0] * n_pairs
        second_labels = [1] * n_pairs

    if pair_id_col and pair_id_col in df:
        raw_pair_ids = df[pair_id_col].tolist()
    else:
        raw_pair_ids = list(range(n_pairs))
    return PairedDataset(
        positive_texts=[text + positive_suffix for text in base_texts],
        negative_texts=[text + negative_suffix for text in base_texts],
        labels=pd.Series(first_labels + second_labels, dtype=int),
        pair_ids=pd.Series(raw_pair_ids + raw_pair_ids),
        opposite_indices=pd.Series(list(range(n_pairs, 2 * n_pairs)) + list(range(n_pairs))),
    )


def _load_polarity_raw(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    positive_suffix: str,
    negative_suffix: str,
) -> PairedDataset:
    _require_columns(df, [text_col])
    if len(df) % 2 != 0:
        raise ValueError("polarity_raw format expects an even number of rows")

    midpoint = len(df) // 2
    labels = df[label_col] if label_col in df else pd.Series([0] * len(df), name=label_col)
    return PairedDataset(
        positive_texts=(df[text_col].astype(str) + positive_suffix).tolist(),
        negative_texts=(df[text_col].astype(str) + negative_suffix).tolist(),
        labels=labels.astype(int).reset_index(drop=True),
        pair_ids=pd.Series(list(range(midpoint)) + list(range(midpoint))),
        opposite_indices=pd.Series(list(range(midpoint, len(df))) + list(range(midpoint))),
    )


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
