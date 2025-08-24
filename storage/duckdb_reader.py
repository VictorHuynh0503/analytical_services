import duckdb
import pandas as pd
from typing import Optional, Dict, Union, List

def read_from_duckdb(
    db_path: str,
    query: str,
    query_params: Optional[Dict[str, Union[str, int, float]]] = None,
    return_type: str = "dataframe"  # or "dict"
) -> Union[pd.DataFrame, List[Dict]]:
    """
    Reads data from a DuckDB file using SQL.

    Args:
        db_path: Path to the DuckDB file
        query: SQL SELECT query (with optional named placeholders like :symbol)
        query_params: Dict of params to fill in SQL placeholders
        return_type: 'dataframe' or 'dict'

    Returns:
        A Pandas DataFrame or List of dicts depending on return_type
    """
    con = duckdb.connect(str(db_path))

    if query_params:
        result = con.execute(query, query_params).fetchdf()
    else:
        result = con.execute(query).fetchdf()

    con.close()

    if return_type == "dict":
        return result.to_dict(orient="records")
    return result
