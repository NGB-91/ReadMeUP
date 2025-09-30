import pandas as pd
import numpy as np
import os
import re
from functools import lru_cache
from app.utils.language import lang_to_iso, iso_to_display
from app.utils.sorting import sort_lns_iterable

@lru_cache(maxsize=1)
def load_df(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure all expected columns exist
    expected_cols = ["title", "author", "rating", "description", "genres", "language", "publisher", "coverImg"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Convert rating to numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Fill missing values and convert to string
    text_cols = ["title", "author", "description", "genres", "language", "publisher", "coverImg"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    # Clean genre strings
    df["genres"] = (
        df["genres"]
        .str.replace("[", "", regex=False)       # remove opening bracket
        .str.replace("]", "", regex=False)       # remove closing bracket
        .str.replace("'", "", regex=False)       # remove single quotes
        .str.replace('"', "", regex=False)       # remove double quotes
        .str.replace(";", ",")                   # unify separators
        .str.replace("|", ",")
        .str.replace("  ", " ")                  # collapse double spaces
        .str.strip()
    )

    # Normalize language codes
    df["language_raw"] = df["language"].astype(str)
    df["language_code"] = df["language_raw"].apply(lang_to_iso)
    df["language"] = df["language_code"].apply(iso_to_display)

    # Extract unique values for filters
    all_genres = sort_lns_iterable({g.strip() for s in df["genres"] if s for g in s.split(",") if g.strip()})
    titles = sort_lns_iterable(set(df["title"]))
    authors = sort_lns_iterable(set(df["author"]))

    return df, all_genres, titles, authors


def apply_multi_filter(
    df: pd.DataFrame,
    title_kw=None,
    author_kw=None,
    genre_kw=None,
    lang_kw=None,
    pub_kw=None,
    shuffle: bool = True,
    seed: int | None = None,
    limit: int | None = None,
):
    filtered_df = df.copy()

    # Apply filters
    if title_kw:
        filtered_df = filtered_df[filtered_df["title"].str.contains(title_kw, case=False, na=False)]
    if author_kw:
        filtered_df = filtered_df[filtered_df["author"].str.contains(author_kw, case=False, na=False)]
    if genre_kw:
        genre_pattern = "|".join([re.escape(g.strip()) for g in genre_kw])
        filtered_df = filtered_df[filtered_df["genres"].str.contains(genre_pattern, case=False, na=False)]
    if lang_kw:
        filtered_df = filtered_df[filtered_df["language"].str.contains(lang_kw, case=False, na=False)]
    if pub_kw:
        filtered_df = filtered_df[filtered_df["publisher"].str.contains(pub_kw, case=False, na=False)]

    # Shuffle and limit results
    if shuffle and not filtered_df.empty:
        filtered_df = filtered_df.sample(frac=1, random_state=seed)
    if limit is not None:
        filtered_df = filtered_df.head(limit)

    return filtered_df.reset_index(drop=True)
