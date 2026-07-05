"""Per-column VALUE filters for st.dataframe tables (the pretty view stays).

`filter_ui(df, key)` renders one control per column inside a collapsed
"🔎 Filtros por columna" expander and returns the filtered frame:
  - numeric columns  → range slider (percent columns shown as 50–100%)
  - low-cardinality  → multiselect of values
  - anything else    → contains-text search
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

PCT_COLS = {"prob", "🤖", "umbral", "acierto_hist", "hist", "meta"}


def filter_ui(df: pd.DataFrame, key: str, skip: tuple = ()) -> pd.DataFrame:
    if df.empty:
        return df
    with st.expander("🔎 Filtros por columna (ej. 🤖 ≥ 80%)"):
        cols = st.columns(3)
        i = 0
        for c in list(df.columns):
            if c in skip or df.empty:
                continue
            w = cols[i % 3]
            i += 1
            s = df[c]
            if c in PCT_COLS and pd.api.types.is_numeric_dtype(s):
                lo, hi = float(s.min() * 100), float(s.max() * 100)
                if lo < hi:
                    a, b = w.slider(f"{c} %", 0, 100, (int(lo), min(100, int(hi) + 1)),
                                    key=f"{key}_{c}")
                    df = df[(s * 100 >= a) & (s * 100 <= b)]
            elif pd.api.types.is_numeric_dtype(s):
                lo, hi = float(s.min()), float(s.max())
                if lo < hi:
                    a, b = w.slider(c, lo, hi, (lo, hi), key=f"{key}_{c}")
                    df = df[s.between(a, b)]
            elif s.nunique(dropna=True) <= 25:
                opts = sorted(s.dropna().astype(str).unique().tolist())
                sel = w.multiselect(c, opts, key=f"{key}_{c}", placeholder="Todos")
                if sel:
                    df = df[s.astype(str).isin(sel)]
            else:
                q = w.text_input(f"🔍 {c}", key=f"{key}_{c}", placeholder="contiene…")
                if q:
                    df = df[s.astype(str).str.contains(q, case=False, na=False)]
            s = None
    return df
