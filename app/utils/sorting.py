import unicodedata
import pandas as pd

def _lns_rank(s: str):
    """
    Returns (group, normalized_key) where:
    group: 0 = letter, 1 = number, 2 = symbol, 3 = empty.
    """
    s = "" if s is None else str(s).strip()
    if not s:
        return (3, "")
    ch = s[0]
    group = 0 if ch.isalpha() else (1 if ch.isdigit() else 2)
    s_norm = unicodedata.normalize("NFKD", s)
    s_norm = "".join(c for c in s_norm if not unicodedata.combining(c))
    return (group, s_norm.casefold())

def sort_lns_iterable(iterable):
    """Sorts items by: letters → numbers → symbols."""
    return sorted(iterable, key=_lns_rank)

def sort_df_by_title_lns(df: pd.DataFrame) -> pd.DataFrame:
    """Sorts a DataFrame by title with priority: letters → numbers → symbols."""
    tmp = df.copy()
    keys = tmp["title"].apply(_lns_rank)
    tmp["__grp"] = keys.apply(lambda t: t[0])
    tmp["__norm"] = keys.apply(lambda t: t[1])
    tmp = tmp.sort_values(by=["__grp", "__norm", "title"], ascending=[True, True, True])
    return tmp.drop(columns=["__grp", "__norm"])
