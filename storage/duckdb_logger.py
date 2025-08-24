import duckdb
import pandas as pd
from typing import List, Dict

def log_df_to_duckdb(
    db_path: str,
    table_name: str,
    df: pd.DataFrame,
    mode: str = "append",  # or "upsert"
    upsert_keys: List[str] = None
):
    if df.empty:
        return

    con = duckdb.connect(str(db_path))

    # Step 1: Register dataframe as a temporary DuckDB view
    con.register("df_view", df)

    # Step 2: Ensure the destination table exists
    con.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" AS SELECT * FROM df_view LIMIT 0;')

    # Step 3: Append or Upsert
    if mode == "append":
        con.execute(f'INSERT INTO "{table_name}" SELECT * FROM df_view;')

    elif mode == "upsert":
        if not upsert_keys:
            raise ValueError("Upsert mode requires upsert_keys.")

        temp_table = f"tmp_{table_name}"
        con.execute(f'CREATE OR REPLACE TEMP TABLE "{temp_table}" AS SELECT * FROM df_view;')

        # Compose the match condition
        match_conditions = " AND ".join(
            [f'"{table_name}"."{col}" = "{temp_table}"."{col}"' for col in upsert_keys]
        )

        # DELETE matching rows from the original table
        delete_query = f'''
        DELETE FROM "{table_name}"
        USING "{temp_table}"
        WHERE {match_conditions};
        '''
        con.execute(delete_query)

        # INSERT new data
        con.execute(f'INSERT INTO "{table_name}" SELECT * FROM "{temp_table}";')

    con.close()

def log_to_duckdb(
    db_path: str,
    table_name: str,
    schema: Dict[str, str],
    data: List[Dict],
    mode: str = "append",  # "append" or "upsert"
    upsert_keys: List[str] = None
):
    """
    Write or update logs to DuckDB with proper quoting for identifiers.

    Args:
        db_path: path to the DuckDB database file
        table_name: name of the table to write
        schema: dictionary like {"col_name": "TEXT", "price": "DOUBLE"}
        data: list of rows as dictionaries
        mode: "append" or "upsert"
        upsert_keys: columns to match for upsert (if mode is "upsert")
    """
    if not data:
        return

    con = duckdb.connect(db_path)

    # ✅ Safe schema definition with quoted column names
    schema_sql = ", ".join(f'"{col}" {dtype}' for col, dtype in schema.items())
    con.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({schema_sql});')

    # ✅ Build column names and placeholders
    columns = list(schema.keys())
    quoted_columns = [f'"{col}"' for col in columns]
    placeholders = ", ".join(["?"] * len(columns))
    col_names_sql = ", ".join(quoted_columns)

    for row in data:
        values = tuple(row.get(col) for col in columns)

        if mode == "append":
            con.execute(
                f'INSERT INTO "{table_name}" ({col_names_sql}) VALUES ({placeholders});',
                values
            )

        elif mode == "upsert" and upsert_keys:
            # ✅ Quoted keys
            quoted_upsert_keys = [f'"{key}"' for key in upsert_keys]
            match = " AND ".join(f'{key}=?' for key in quoted_upsert_keys)
            upsert_values = tuple(row.get(k) for k in upsert_keys)

            # Check if row exists
            exists = con.execute(
                f'SELECT COUNT(*) FROM "{table_name}" WHERE {match};',
                upsert_values
            ).fetchone()[0]

            if exists:
                # UPDATE path
                update_cols = [col for col in columns if col not in upsert_keys]
                quoted_update = ", ".join(f'"{col}"=?' for col in update_cols)
                update_values = tuple(row.get(col) for col in update_cols)

                con.execute(
                    f'UPDATE "{table_name}" SET {quoted_update} WHERE {match};',
                    update_values + upsert_values
                )
            else:
                # INSERT path
                con.execute(
                    f'INSERT INTO "{table_name}" ({col_names_sql}) VALUES ({placeholders});',
                    values
                )

    con.close()

def df_to_duckdb_schema(df: pd.DataFrame) -> dict:
    dtype_map = {
        "object": "TEXT",
        "int64": "BIGINT",
        "float64": "DOUBLE",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP"
    }

    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        duck_type = dtype_map.get(dtype, "TEXT")  # fallback to TEXT
        schema[col] = duck_type

    return schema