from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import numpy as np

@dataclass
class BuiltDataset:
    """
    Container for model-ready datasets.

    Attributes
    ----------
    X:
        Model input features.

        For tabular models:
            shape = [n_samples, n_features]

        For sequence models:
            shape = [n_samples, seq_len, n_features]

    y:
        Target values, shape = [n_samples]

    meta:
        Metadata aligned with X and y.
        It must contain at least date_col and stock_col.
        Additional columns, such as industry_code or price, can also be kept.
    """

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame

class PanelDatasetBuilder:
    """
    Build model-ready datasets from a clean stock panel DataFrame.

    This builder does not read files, preprocess factors, standardize features,
    construct targets, or perform backtesting. Its only responsibility is to
    convert a clean panel DataFrame into X / y / meta for tabular and sequence
    models.

    Expected input DataFrame format:
        date_col | stock_col | feature columns | target_col | optional meta columns

    Examples
    --------
    Tabular output:
        X:    [n_samples, n_features]
        y:    [n_samples]
        meta: [n_samples, meta_columns]

    Sequence output:
        X:    [n_samples, seq_len, n_features]
        y:    [n_samples]
        meta: [n_samples, meta_columns]
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        target_col: str,
        date_col: str = "date",
        stock_col: str = "stock_id",
        seq_len: int = 60,
        meta_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col
        self.stock_col = stock_col
        self.seq_len = seq_len
        self.meta_cols = meta_cols if meta_cols is not None else []

        self._validate_config()

        # Always keep date and stock identifiers in meta.
        # Extra meta columns are appended while preserving order and removing duplicates.

        self.output_meta_cols = self._deduplicate_preserve_order(
            [self.date_col, self.stock_col] + self.meta_cols
        )

    @staticmethod
    def _deduplicate_preserve_order(values: Sequence[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result
    
    def _validate_config(self) -> None:
        if len(self.feature_cols) == 0:
            raise ValueError("feature_cols must not be empty")

        if len(set(self.feature_cols)) != len(self.feature_cols):
            raise ValueError("feature_cols contains duplicated column names")

        if not isinstance(self.target_col, str) or self.target_col == "":
            raise ValueError("target_col must be a non-empty string")

        if not isinstance(self.date_col, str) or self.date_col == "":
            raise ValueError("date_col must be a non-empty string")

        if not isinstance(self.stock_col, str) or self.stock_col == "":
            raise ValueError("stock_col must be a non-empty string")

        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")

        if self.target_col in self.feature_cols:
            raise ValueError("target_col must not be included in feature_cols")

        if self.date_col in self.feature_cols:
            raise ValueError("date_col must not be included in feature_cols")

        if self.stock_col in self.feature_cols:
            raise ValueError("stock_col must not be included in feature_cols")

        if len(set(self.meta_cols)) != len(self.meta_cols):
            raise ValueError("meta_cols contains duplicated column names")

        if self.target_col in self.meta_cols:
            raise ValueError("target_col should not be included in meta_cols")
        
    def _required_cols(self) -> List[str]:
        return self._deduplicate_preserve_order(
            [self.date_col, self.stock_col] + self.feature_cols + [self.target_col] + self.meta_cols
        )
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_cols = [col for col in self._required_cols() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    def _validate_numeric_feature_target(self, prepared: pd.DataFrame) -> None:
        non_numeric_features = [
            col
            for col in self.feature_cols
            if not pd.api.types.is_numeric_dtype(prepared[col])
        ]
        if non_numeric_features:
            raise ValueError(
                f"All feature columns must be numeric. "
                f"Non-numeric feature columns: {non_numeric_features}"
            )

        if not pd.api.types.is_numeric_dtype(prepared[self.target_col]):
            raise ValueError("Target column must be numeric.")

    def _resolve_output_meta_cols(
        self,
        df: pd.DataFrame,
        output_meta_cols: Optional[Sequence[str]],
    ) -> List[str]:
        if output_meta_cols is None:
            resolved_cols = list(self.output_meta_cols)
        else:
            resolved_cols = self._deduplicate_preserve_order(
                [self.date_col, self.stock_col] + list(output_meta_cols)
            )

        missing_cols = [col for col in resolved_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing output meta columns: {missing_cols}"
            )

        return resolved_cols
        
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Copy, validate, convert date column, check duplicates, and sort panel data.

        This method intentionally does not fill missing values or standardize features.
        Missing / infinite values are handled when building tabular or sequence datasets.
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected a DataFrame, but got {type(df)}")
        
        self._validate_columns(df)

        prepared = df.copy()
        prepared[self.date_col] = pd.to_datetime(prepared[self.date_col], errors="raise")

        duplicated_mask = prepared.duplicated(
            subset=[self.date_col, self.stock_col],
            keep=False
        )
        if duplicated_mask.any():
            duplicated_count = int(duplicated_mask.sum())
            raise ValueError(
                "Found duplicated stock-date rows. "
                f"Duplicated row count: {duplicated_count}. "
                "Please fix duplicates in preprocess.py before building datasets."
            )
        
        prepared = prepared.sort_values(
            [self.stock_col, self.date_col],
            kind = 'mergesort'
        ).reset_index(drop=True)

        return prepared
    
    def build_tabular_dataset(
        self,
        df: pd.DataFrame,
        output_meta_cols: Optional[Sequence[str]] = None,
    ) -> BuiltDataset:
        """
        Build a tabular dataset for models such as XGBoost and LightGBM.

        Each valid row is treated as one sample.

        Output shapes:
            X:    [n_samples, n_features]
            y:    [n_samples]
            meta: [n_samples, output_meta_cols]

        Rows with NaN or Inf in feature columns or target column are dropped.
        This method does not fill missing values or modify feature values.
        """

        prepared = self._prepare_dataframe(df)
        resolved_meta_cols = self._resolve_output_meta_cols(prepared, output_meta_cols)
        self._validate_numeric_feature_target(prepared)
        
        X_all = prepared[self.feature_cols].to_numpy(dtype=np.float32, copy = True)
        y_all = prepared[self.target_col].to_numpy(dtype=np.float32, copy = True).reshape(-1)

        valid_mask = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)

        X = X_all[valid_mask]
        y = y_all[valid_mask]
        meta = prepared.loc[valid_mask, resolved_meta_cols].reset_index(drop=True)

        if X.shape[0] == 0:
            raise ValueError(
                "No valid tabular samples were built. "
                "Please check missing or infinite values in feature and target columns."
            )

        return BuiltDataset(X=X, y=y, meta=meta)
    
    def build_sequence_dataset(
        self,
        df: pd.DataFrame,
        output_meta_cols: Optional[Sequence[str]] = None,
    ) -> BuiltDataset:
        """
        Build a sequence dataset for models such as DLinear, TSMixer, PatchTST,
        and iTransformer.

        For each stock, a rolling window is constructed independently.

        For a sample ending at date t:
            X = features from t - seq_len + 1 to t
            y = target at t
            meta = metadata at t

        Output shapes:
            X:    [n_samples, seq_len, n_features]
            y:    [n_samples]
            meta: [n_samples, output_meta_cols]

        A sample is skipped if its feature window or target contains NaN or Inf.
        This method does not fill missing values or modify feature values.
        """

        prepared = self._prepare_dataframe(df)
        resolved_meta_cols = self._resolve_output_meta_cols(prepared, output_meta_cols)
        self._validate_numeric_feature_target(prepared)
        
        X_list = []
        y_list = []
        meta_rows = []

        for _, group in prepared.groupby(self.stock_col):
            group = group.sort_values(self.date_col, kind="mergesort").reset_index(drop=True)

            if len(group) < self.seq_len:
                continue

            features = group[self.feature_cols].to_numpy(dtype=np.float32, copy = True)
            targets = group[self.target_col].to_numpy(dtype=np.float32, copy = True).reshape(-1)
            meta_group = group[resolved_meta_cols]

            for end_idx in range(self.seq_len - 1, len(group)):
                start_idx = end_idx - self.seq_len + 1
                X_window = features[start_idx:end_idx + 1]
                y_value = targets[end_idx]

                if not np.isfinite(X_window).all():
                    continue

                if not np.isfinite(y_value):
                    continue

                X_list.append(X_window)
                y_list.append(y_value)
                meta_rows.append(meta_group.iloc[end_idx])

        if len(X_list) == 0:
            raise ValueError(
                "No valid sequence samples were built. "
                "Please check seq_len, panel history length, and missing or infinite values."
            )

        X = np.stack(X_list, axis=0).astype(np.float32, copy = False)
        y = np.array(y_list, dtype=np.float32).reshape(-1)
        meta = pd.DataFrame(meta_rows).reset_index(drop=True)

        return BuiltDataset(X=X, y=y, meta=meta)
        
    @staticmethod
    def _subset_built_dataset(dataset: BuiltDataset, mask: np.ndarray) -> BuiltDataset:
        """
        Subset a BuiltDataset with a boolean mask while keeping X, y, and meta aligned.
        """

        if mask.ndim != 1:
            raise ValueError(f"mask must be 1D, got shape {mask.shape}")

        if dataset.X.shape[0] != mask.shape[0]:
            raise ValueError(
                f"X and mask length mismatch: {dataset.X.shape[0]} vs {mask.shape[0]}"
            )

        if dataset.y.shape[0] != mask.shape[0]:
            raise ValueError(
                f"y and mask length mismatch: {dataset.y.shape[0]} vs {mask.shape[0]}"
            )

        if len(dataset.meta) != mask.shape[0]:
            raise ValueError(
                f"meta and mask length mismatch: {len(dataset.meta)} vs {mask.shape[0]}"
            )
        
        return BuiltDataset(
            X=dataset.X[mask].copy(),
            y=dataset.y[mask].copy(),
            meta=dataset.meta.loc[mask].reset_index(drop=True).copy(),
        )

    def split_built_dataset(
        self,
        dataset: BuiltDataset,
        train_end: str,
        valid_end: str,
    ) -> Tuple[BuiltDataset, BuiltDataset, BuiltDataset]:
        """
        Split an already-built dataset into train / valid / test by sample date.

        The split is based on dataset.meta[date_col].

        Split rule:
            train: date <= train_end
            valid: train_end < date <= valid_end
            test:  date > valid_end

        This method works for both tabular and sequence datasets.
        For sequence datasets, the sample date is the window end date.
        """

        if not isinstance(dataset, BuiltDataset):
            raise TypeError(f"dataset must be a BuiltDataset, got {type(dataset)}")

        if self.date_col not in dataset.meta.columns:
            raise ValueError(
                f"dataset.meta must contain date_col '{self.date_col}' for date split"
            )

        n_samples = dataset.X.shape[0]
        if dataset.y.shape[0] != n_samples:
            raise ValueError(
                f"X and y length mismatch: {n_samples} vs {dataset.y.shape[0]}"
            )

        if len(dataset.meta) != n_samples:
            raise ValueError(
                f"X and meta length mismatch: {n_samples} vs {len(dataset.meta)}"
            )

        sample_dates = pd.to_datetime(dataset.meta[self.date_col], errors="raise")
        train_end_dt = pd.to_datetime(train_end, errors="raise")
        valid_end_dt = pd.to_datetime(valid_end, errors="raise")

        if train_end_dt >= valid_end_dt:
            raise ValueError(
                f"train_end must be earlier than valid_end, got {train_end} and {valid_end}"
            )

        train_mask = (sample_dates <= train_end_dt).to_numpy()
        valid_mask = ((sample_dates > train_end_dt) & (sample_dates <= valid_end_dt)).to_numpy()
        test_mask = (sample_dates > valid_end_dt).to_numpy()

        split_counts = {
            "train": int(train_mask.sum()),
            "valid": int(valid_mask.sum()),
            "test": int(test_mask.sum()),
        }
        empty_splits = [name for name, count in split_counts.items() if count == 0]
        if empty_splits:
            raise ValueError(
                f"Empty split(s) after date split: {empty_splits}. "
                f"Split counts: {split_counts}. "
                "Please check train_end, valid_end, and sample date range."
            )

        train_data = self._subset_built_dataset(dataset, train_mask)
        valid_data = self._subset_built_dataset(dataset, valid_mask)
        test_data = self._subset_built_dataset(dataset, test_mask)

        return train_data, valid_data, test_data
    
    def build_tabular_splits(
        self,
        df: pd.DataFrame,
        train_end: str,
        valid_end: str,
    ) -> Tuple[BuiltDataset, BuiltDataset, BuiltDataset]:
        """
        Build a tabular dataset from a full DataFrame and split it by date.

        This is the recommended shortcut for tabular models such as XGBoost
        and LightGBM when a full clean panel DataFrame is available.
        """

        dataset = self.build_tabular_dataset(df)
        return self.split_built_dataset(
            dataset=dataset,
            train_end=train_end,
            valid_end=valid_end,
        )

    def build_sequence_splits(
        self,
        df: pd.DataFrame,
        train_end: str,
        valid_end: str,
    ) -> Tuple[BuiltDataset, BuiltDataset, BuiltDataset]:
        """
        Build a sequence dataset from a full DataFrame and split it by date.

        This is the recommended shortcut for sequence models such as DLinear,
        TSMixer, PatchTST, and iTransformer.

        The sequence dataset is built before splitting. Therefore, validation
        and test samples can use historical observations from earlier periods,
        while each sample's window still ends at its own sample date.
        """

        dataset = self.build_sequence_dataset(df)
        return self.split_built_dataset(
            dataset=dataset,
            train_end=train_end,
            valid_end=valid_end,
        )

    @staticmethod
    def _make_temporary_split_col(dfs: Sequence[pd.DataFrame]) -> str:
        base_col = "__split_label__"
        existing_cols = set()
        for df in dfs:
            existing_cols.update(df.columns)

        split_col = base_col
        counter = 1
        while split_col in existing_cols:
            split_col = f"{base_col}_{counter}"
            counter += 1

        return split_col

    def _concat_frames_with_split_label(
        self,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, str]:
        """
        Concatenate pre-split frames and add a temporary split label column.
        """

        if not isinstance(train_df, pd.DataFrame):
            raise TypeError(f"train_df must be a pandas DataFrame, got {type(train_df)}")

        input_frames = [("train", train_df)]

        if valid_df is not None:
            if not isinstance(valid_df, pd.DataFrame):
                raise TypeError(f"valid_df must be a pandas DataFrame, got {type(valid_df)}")
            input_frames.append(("valid", valid_df))

        if test_df is not None:
            if not isinstance(test_df, pd.DataFrame):
                raise TypeError(f"test_df must be a pandas DataFrame, got {type(test_df)}")
            input_frames.append(("test", test_df))

        if len(input_frames) == 1:
            raise ValueError("At least one of valid_df or test_df must be provided")

        split_col = self._make_temporary_split_col([frame for _, frame in input_frames])

        labeled_frames = []
        for split_name, frame in input_frames:
            labeled = frame.copy()
            labeled[split_col] = split_name
            labeled_frames.append(labeled)

        combined = pd.concat(labeled_frames, axis=0, ignore_index=True)

        return combined, split_col
    
    def _split_built_dataset_by_label(
        self,
        dataset: BuiltDataset,
        split_col: str,
    ) -> tuple[BuiltDataset, Optional[BuiltDataset], Optional[BuiltDataset]]:
        """
        Split a BuiltDataset by a temporary split label stored in meta.
        """

        if split_col not in dataset.meta.columns:
            raise ValueError(f"dataset.meta must contain temporary split column: {split_col}")

        split_values = dataset.meta[split_col].astype(str)

        def subset_or_none(split_name: str) -> Optional[BuiltDataset]:
            mask = (split_values == split_name).to_numpy()
            if int(mask.sum()) == 0:
                return None

            subset = self._subset_built_dataset(dataset, mask)
            subset.meta = subset.meta.drop(columns=[split_col]).reset_index(drop=True)
            return subset

        train_data = subset_or_none("train")
        valid_data = subset_or_none("valid")
        test_data = subset_or_none("test")

        if train_data is None:
            raise ValueError("No valid train samples were built from train_df")

        return train_data, valid_data, test_data
    
    def build_tabular_splits_from_frames(
        self,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> tuple[BuiltDataset, Optional[BuiltDataset], Optional[BuiltDataset]]:
        """
        Build tabular splits from already-separated train / valid / test frames.

        This compatibility method is useful when data is stored in separate files.
        It temporarily concatenates the frames, builds one tabular dataset, and
        then splits the built dataset back by the original frame labels.
        """

        combined, split_col = self._concat_frames_with_split_label(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )

        extended_meta_cols = self._deduplicate_preserve_order(
            self.output_meta_cols + [split_col]
        )
        dataset = self.build_tabular_dataset(
            combined,
            output_meta_cols=extended_meta_cols,
        )

        return self._split_built_dataset_by_label(dataset, split_col)

    def build_sequence_splits_from_frames(
        self,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> tuple[BuiltDataset, Optional[BuiltDataset], Optional[BuiltDataset]]:
        """
        Build sequence splits from already-separated train / valid / test frames.

        This compatibility method is useful when data is stored in separate files.
        The frames are temporarily concatenated before sequence construction, so
        validation and test samples can use historical observations from earlier
        frames while still being assigned back to their original split labels.
        """

        combined, split_col = self._concat_frames_with_split_label(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )

        extended_meta_cols = self._deduplicate_preserve_order(
            self.output_meta_cols + [split_col]
        )
        dataset = self.build_sequence_dataset(
            combined,
            output_meta_cols=extended_meta_cols,
        )

        return self._split_built_dataset_by_label(dataset, split_col)

if __name__ == "__main__":
    # Minimal smoke test for both tabular and sequence dataset construction.
    rng = np.random.default_rng(42)

    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    stock_ids = ["A", "B", "C"]

    rows = []
    for stock_id in stock_ids:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "stock_id": stock_id,
                    "factor_1": rng.normal(),
                    "factor_2": rng.normal(),
                    "target": rng.normal(),
                    "industry_code": "IND_1" if stock_id in {"A", "B"} else "IND_2",
                }
            )

    df = pd.DataFrame(rows)

    builder = PanelDatasetBuilder(
        feature_cols=["factor_1", "factor_2"],
        target_col="target",
        date_col="date",
        stock_col="stock_id",
        seq_len=5,
        meta_cols=["industry_code"],
    )

    tabular_train, tabular_valid, tabular_test = builder.build_tabular_splits(
        df=df,
        train_end="2020-01-05",
        valid_end="2020-01-08",
    )

    sequence_train, sequence_valid, sequence_test = builder.build_sequence_splits(
        df=df,
        train_end="2020-01-05",
        valid_end="2020-01-08",
    )

    train_df = df[df["date"] <= "2020-01-05"].copy()
    valid_df = df[(df["date"] > "2020-01-05") & (df["date"] <= "2020-01-08")].copy()
    test_df = df[df["date"] > "2020-01-08"].copy()

    frame_train, frame_valid, frame_test = builder.build_sequence_splits_from_frames(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
    )

    print("Tabular train X shape:", tabular_train.X.shape)
    print("Tabular valid X shape:", tabular_valid.X.shape)
    print("Tabular test X shape:", tabular_test.X.shape)
    print("Sequence train X shape:", sequence_train.X.shape)
    print("Sequence valid X shape:", sequence_valid.X.shape)
    print("Sequence test X shape:", sequence_test.X.shape)
    print("Frame-based sequence train X shape:", frame_train.X.shape)
    print("Frame-based sequence valid X shape:", None if frame_valid is None else frame_valid.X.shape)
    print("Frame-based sequence test X shape:", None if frame_test is None else frame_test.X.shape)