"""
Extra statistics for run 20261124_183727 (numbers quoted in the thesis, not printed by v7).
Reads D_lst, the context workbook and the exported recommendations; prints aggregates only.
Run from the project root: python diagnostics/eval_fix_20261124/extra_stats.py
"""
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import elderly_recsys_tools as erst

RUN = "20261124_183727"
OLD_RUN = "20261015_150006"
K = 5
recs_path = ROOT / "outputs" / "heldout_evaluation" / "recommendations"


def load_recs(name):
    df = pd.read_excel(recs_path / name)
    recs = defaultdict(set)
    for uid, item in zip(df["uID"], df["ActivitySequence"]):
        recs[uid].add(erst.normalize_item(json.loads(item)))
    return df, recs


def top_ref(pairs):
    return set(i for i, _ in sorted(pairs, key=lambda x: (-x[1], x[0]))[:K])


def f1(predicted, actual):
    return 2 * len(predicted & actual) / (len(predicted) + len(actual))


# Same held-out split as v7
with open(ROOT / "data" / "D_lst" / "D_lst_full_20260202_150312.pkl", "rb") as f:
    D_lst = pickle.load(f)
_, heldout_idx = next(ShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(np.zeros(len(D_lst))))
by_user = defaultdict(list)
for i in heldout_idx:
    by_user[D_lst[i][0]].append((erst.normalize_item(D_lst[i][1]), float(D_lst[i][3])))
users = sorted(by_user)

# Same context sampling as v7 (pool from the recommendable activities, seed 42)
action_df = pd.read_excel(ROOT / "data" / "context" / "ActivityContextGen_v12.xlsx", sheet_name="ActionLst").replace(-1, np.nan)
ctx_cols = [f"act_C_{c}{k}" for c in "TP" for k in (1, 2, 3)]
actID_context_dc = {row["actID"]: {c: row[c] for c in ctx_cols} for _, row in action_df.iterrows()}
actIDs = sorted({a for u in users for item, _ in by_user[u] for a in item})
C_T_pool, C_P_pool = erst.build_context_pool({a: actID_context_dc[a] for a in actIDs})
random.seed(42)
context = {u: erst.sample_random_context(C_T_pool, C_P_pool) for u in users}

df_42, recs_42 = load_recs(f"recommendations_4_2_with_PQ_{RUN}.xlsx")
mismatch = sum((context[u]["C_T"], context[u]["C_P"]) != (t, p)
               for u, t, p in zip(df_42["uID"], df_42["C_T"], df_42["C_P"]))

feasible = {u: [(i, r) for i, r in by_user[u]
                if erst.item_context_feasible(i, context[u], actID_context_dc)]
            for u in users}
n = {u: len(feasible[u]) for u in users}
none = [u for u in users if n[u] == 0]
small = [u for u in users if 1 <= n[u] <= K]
large = [u for u in users if n[u] > K]
svd = {u: f1(recs_42[u], top_ref(feasible[u])) for u in small + large}
# Expected F1 of a random order: n <= K returns the whole reference (1), otherwise K / n
rnd = {u: min(1.0, K / n[u]) for u in small + large}

print(f"Run: {RUN} | users: {len(users)} | test ratings: {len(heldout_idx)}")
print(f"Recommendable activities: {len(actIDs)} | |C_T|={len(C_T_pool)} |C_P|={len(C_P_pool)}")
print(f"Context mismatches against the export: {mismatch}")
print("\n[B, context reference] feasible held-out items per user")
print(f"0 items: {len(none)} | 1-{K} items: {len(small)} ({len(small) / len(users):.3f} of all users) | >{K} items: {len(large)}")
print(f"SVD F1:             all with reference={np.mean(list(svd.values())):.3f} | 1-{K}={np.mean([svd[u] for u in small]):.3f} | >{K}={np.mean([svd[u] for u in large]):.3f}")
print(f"Random expected F1: all with reference={np.mean(list(rnd.values())):.3f} | 1-{K}={np.mean([rnd[u] for u in small]):.3f} | >{K}={np.mean([rnd[u] for u in large]):.3f}")

_, recs_old = load_recs(f"recommendations_4_2_with_PQ_{OLD_RUN}.xlsx")
print("\n[Old B, global reference]")
print(f"Old run {OLD_RUN} F1:         {np.mean([f1(recs_old[u], top_ref(by_user[u])) for u in users]):.4f}")
print(f"Random order, mean of {K}/n_u: {np.mean([K / len(by_user[u]) for u in users]):.4f}")
