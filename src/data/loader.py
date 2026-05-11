from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, Path]

def load_panel_data(
        path: PathLike,
        sheet_name: Optional[Union[str, int]] = None,
        **kwargs,
) -> pd.DataFrame:
    """
    Load panel data from a local file.

    Supported file formats:
        - .parquet / .pq
        - .csv
        - .xlsx / .xls

    This function only reads data into a pandas DataFrame.
    It does not perform preprocessing, date conversion, missing-value handling,
    feature engineering, or target construction.

    Parameters
    ----------
    path:
        Path to the input data file.

    sheet_name:
        Sheet name or sheet index for Excel files.
        If omitted, the first worksheet is loaded.
        Ignored for non-Excel files.

    **kwargs:
        Additional keyword arguments passed to the corresponding pandas reader.

    Returns
    -------
    pd.DataFrame
        Loaded raw panel data.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    ValueError
        If the file format is unsupported, or if the loaded object is not a DataFrame.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    
    suffix = file_path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        if sheet_name is not None:
            raise ValueError("sheet_name is not applicable for Parquet files.")
        df = pd.read_parquet(file_path, **kwargs)
    
    elif suffix == ".csv":
        if sheet_name is not None:
            raise ValueError("sheet_name is not applicable for CSV files.")
        df = pd.read_csv(file_path, **kwargs)

    elif suffix in {".xlsx", ".xls"}:
        excel_sheet_name = 0 if sheet_name is None else sheet_name
        df = pd.read_excel(file_path, sheet_name=excel_sheet_name, **kwargs)

    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a DataFrame, but got {type(df)}")
    
    return df