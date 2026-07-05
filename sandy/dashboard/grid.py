"""Excel-style tables: AgGrid with a value-filter box under EVERY column header
(text contains, number ranges, value sets) + sorting. Shared by all pages."""
from __future__ import annotations

import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

PCT_COLS = {"prob", "🤖", "umbral", "acierto_hist", "hist", "meta"}


def show(df: pd.DataFrame, key: str, height: int = 440) -> None:
    df = df.copy()
    for c in df.columns:
        if c in PCT_COLS and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = (df[c] * 100).round(1)  # numeric % → range filters work (e.g. > 90)
    df.columns = [f"{c} %" if c in PCT_COLS else c for c in df.columns]
    for c in df.columns:  # AgGrid needs JSON-serializable values
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].astype(str)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filter=True, sortable=True, resizable=True,
                                floatingFilter=True, wrapText=False)
    gb.configure_grid_options(domLayout="normal", suppressMenuHide=True)
    AgGrid(df, gridOptions=gb.build(), height=height, key=key,
           fit_columns_on_grid_load=len(df.columns) <= 8,
           theme="streamlit", allow_unsafe_jscode=False)
