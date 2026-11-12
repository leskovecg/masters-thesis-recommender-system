# make_context_group_histograms_v6.py
# ------------------------------------------------------------
# Namen:
# - za ContextG_act_3 / 4 / 5 generira histograme (bar charts)
# - fokus: dominantni čas (T), prostor (P) in spremljevalna aktivnost (A)
# - shrani slike v outputs/latex/figs/ (za Overleaf)
#
# Input:
#   - ActivityContextGen_v12.xlsx (sheet ActionLst)  <-- glavni podatki
#   - ContextGroups_act_3_4_5_v02.xlsx               <-- opcijsko (za referenco)
#
# Output:
#   - PNG histograms: ContextG_act_X__grpY__time/place/comp.png
#   - CSV summaries:  ContextG_act_X__grpY__time_top.csv  ...
# ------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATHS (nastavi)
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

data_path = BASE_DIR / 'data'
context_path = data_path / 'context'
ACTIVITY_PATH = context_path / "ActivityContextGen_v12.xlsx"
GROUPS_PATH   = context_path / "ContextGroups_act_3_4_5_v02.xlsx"  # ni nujno za histograme, samo da je zraven
SHEET_NAME    = "ActionLst"

outputs_path = BASE_DIR / 'outputs'
OUT_DIR = outputs_path / "context_group_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SETTINGS
# =========================
GROUP_COLS = ["ContextG_act_3", "ContextG_act_4", "ContextG_act_5"]

TIME_COLS_REC = ["rec_C_T1", "rec_C_T2", "rec_C_T3"]
PLACE_COLS_REC = ["rec_C_P1", "rec_C_P2", "rec_C_P3"]

TIME_COLS_ACT = ["act_C_T1", "act_C_T2", "act_C_T3"]
PLACE_COLS_ACT = ["act_C_P1", "act_C_P2", "act_C_P3"]

COMP_COLS_ACT = ["act_C_A1", "act_C_A2", "act_C_A3"]  # spremljevalna aktivnost

TOP_K = 12          # koliko kategorij pokažeš na histogramu
MIN_COUNT = 2       # ignoriraj kategorije, ki se pojavijo < MIN_COUNT (da ni šuma)
FIG_DPI = 200


# =========================
# HELPERS
# =========================
def _clean_value(x):
    """Normalizira vrednosti v kategorijah."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() == "nan" or s == "-1":
        return np.nan
    return s

def first_non_null_in_row(row, cols):
    """Vzame prvo ne-NaN vrednost iz listy stolpcev."""
    for c in cols:
        v = _clean_value(row.get(c, np.nan))
        if pd.notna(v):
            return v
    return np.nan

def add_dominant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Doda dominanten čas/prostor/spremljevalno aktivnost za rec+act posebej in skupaj."""
    df = df.copy()

    # dominantni (prva ne-NaN) rec in act
    df["dom_T_rec"] = df.apply(lambda r: first_non_null_in_row(r, TIME_COLS_REC), axis=1)
    df["dom_P_rec"] = df.apply(lambda r: first_non_null_in_row(r, PLACE_COLS_REC), axis=1)

    df["dom_T_act"] = df.apply(lambda r: first_non_null_in_row(r, TIME_COLS_ACT), axis=1)
    df["dom_P_act"] = df.apply(lambda r: first_non_null_in_row(r, PLACE_COLS_ACT), axis=1)

    df["dom_A_act"] = df.apply(lambda r: first_non_null_in_row(r, COMP_COLS_ACT), axis=1)

    # kombinirano: najprej rec, če ni, potem act
    df["dom_T"] = df["dom_T_rec"].combine_first(df["dom_T_act"])
    df["dom_P"] = df["dom_P_rec"].combine_first(df["dom_P_act"])

    return df

def value_counts_top(series: pd.Series, top_k=TOP_K, min_count=MIN_COUNT) -> pd.DataFrame:
    """Vrne tabelo top-k frekvenc za kategorije."""
    s = series.dropna().map(_clean_value).dropna()
    vc = s.value_counts()
    vc = vc[vc >= min_count]
    vc = vc.head(top_k)
    out = vc.reset_index()
    out.columns = ["value", "count"]
    return out

def save_barplot(count_df: pd.DataFrame, title: str, out_path: Path):
    """Shrani bar chart (kategorični histogram)."""
    plt.figure(figsize=(10, 5))
    plt.bar(count_df["value"].astype(str), count_df["count"].astype(int))
    plt.title(title)
    plt.xlabel("Kategorija")
    plt.ylabel("Št. pojavitev")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()

def safe_group_sort_key(x):
    """Stabilen sort key tudi če so grupe int/float/str mešane."""
    return str(x)


# =========================
# MAIN
# =========================
def main():
    print("Loading:", ACTIVITY_PATH)
    df = pd.read_excel(ACTIVITY_PATH, sheet_name=SHEET_NAME)
    df.columns = df.columns.astype(str).str.strip()

    # check group cols
    missing = [c for c in GROUP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manjkajo group stolpci v {ACTIVITY_PATH.name}: {missing}")

    # dodaj dominantne stolpce
    df = add_dominant_columns(df)

    # opcijsko: naložimo groups file samo da preveriš, da je "zraven"
    if GROUPS_PATH.exists():
        print("Found groups reference:", GROUPS_PATH)
    else:
        print("WARNING: groups reference file not found (OK, not required):", GROUPS_PATH)

    # naredi histograme za vsako segmentacijo posebej
    for gcol in GROUP_COLS:
        print(f"\n=== Processing {gcol} ===")

        # grupe v tem stolpcu
        groups = df[gcol].dropna().unique().tolist()
        groups = sorted(groups, key=safe_group_sort_key)

        for grp in groups:
            df_g = df[df[gcol] == grp].copy()
            n = len(df_g)
            if n == 0:
                continue

            prefix = f"{gcol}__grp{str(grp)}"
            print(f"  - group {grp} (N={n})")

            # ===== TIME (dom_T) =====
            time_top = value_counts_top(df_g["dom_T"])
            if len(time_top) > 0:
                time_csv = OUT_DIR / f"{prefix}__time_top.csv"
                time_png = OUT_DIR / f"{prefix}__time.png"
                time_top.to_csv(time_csv, index=False, encoding="utf-8-sig")
                save_barplot(
                    time_top,
                    title=f"{gcol} | grupa {grp} | dominantni čas (dom_T) | N={n}",
                    out_path=time_png
                )

            # ===== PLACE (dom_P) =====
            place_top = value_counts_top(df_g["dom_P"])
            if len(place_top) > 0:
                place_csv = OUT_DIR / f"{prefix}__place_top.csv"
                place_png = OUT_DIR / f"{prefix}__place.png"
                place_top.to_csv(place_csv, index=False, encoding="utf-8-sig")
                save_barplot(
                    place_top,
                    title=f"{gcol} | grupa {grp} | dominantni prostor (dom_P) | N={n}",
                    out_path=place_png
                )

            # ===== COMPANION (dom_A_act) =====
            comp_top = value_counts_top(df_g["dom_A_act"])
            if len(comp_top) > 0:
                comp_csv = OUT_DIR / f"{prefix}__comp_top.csv"
                comp_png = OUT_DIR / f"{prefix}__comp.png"
                comp_top.to_csv(comp_csv, index=False, encoding="utf-8-sig")
                save_barplot(
                    comp_top,
                    title=f"{gcol} | grupa {grp} | spremljevalna aktivnost (dom_A_act) | N={n}",
                    out_path=comp_png
                )

        # dodatno: en “overview” CSV za vse grupe (koliko elementov v vsaki)
        overview = (
            df[gcol]
            .dropna()
            .map(_clean_value)
            .dropna()
            .value_counts()
            .reset_index()
        )
        overview.columns = ["group", "count"]
        overview.to_csv(OUT_DIR / f"{gcol}__overview_counts.csv", index=False, encoding="utf-8-sig")

    print("\nOK ✅ Histograms + CSV summaries saved to:")
    print(str(OUT_DIR))


if __name__ == "__main__":
    main()
