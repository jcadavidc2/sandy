"""Chronological train/validation split for Sandy.

Task 10.1: chronological_split() splits a DataFrame by game date so that
the most recent 15% of games form the validation set and the rest form
the training set.

Splitting by game (not by row) prevents same-game data leakage — all rows
from a given game_pk go entirely to one split.

Pure function: no I/O, no randomness, deterministic given the same input.

Requirements: 6.2
"""
from __future__ import annotations

import math
from typing import NamedTuple

import pandas as pd


class SplitResult(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    n_train_games: int
    n_val_games: int


def chronological_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
) -> SplitResult:
    """Split *df* chronologically into train and validation sets.

    Algorithm:
    1. Extract unique (game_date, game_pk) pairs and sort ascending.
    2. The last ``ceil(n_games * val_fraction)`` games by game_date go to
       validation; everything earlier goes to training.
    3. Assign each row to train or val based on its game_pk.

    Parameters
    ----------
    df:           DataFrame with at least ``game_pk`` and ``game_date`` columns.
    val_fraction: Fraction of games (by count) to use for validation.
                  Default 0.15 = 15%.

    Returns
    -------
    SplitResult with train/val DataFrames and game counts.

    Requirements: 6.2
    """
    if df.empty:
        return SplitResult(
            train=df.copy(),
            val=df.copy(),
            n_train_games=0,
            n_val_games=0,
        )

    # Step 1: unique (game_date, game_pk) sorted ascending
    game_dates = (
        df[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk"], ascending=True)
        .reset_index(drop=True)
    )

    n_games = len(game_dates)
    n_val = max(1, math.ceil(n_games * val_fraction))
    n_train = n_games - n_val

    # Step 2: split game_pk sets
    train_pks = set(game_dates.iloc[:n_train]["game_pk"].tolist())
    val_pks = set(game_dates.iloc[n_train:]["game_pk"].tolist())

    # Step 3: assign rows
    train_df = df[df["game_pk"].isin(train_pks)].copy()
    val_df = df[df["game_pk"].isin(val_pks)].copy()

    return SplitResult(
        train=train_df,
        val=val_df,
        n_train_games=n_train,
        n_val_games=n_val,
    )


__all__ = ["SplitResult", "chronological_split"]
