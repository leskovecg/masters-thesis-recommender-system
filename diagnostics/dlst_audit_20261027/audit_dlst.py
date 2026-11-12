"""
Neodvisna revizija D_lst (diagnostika, NE del produkcijskega postopka).

Ne spreminja produkcijske kode, vhodov, obstoječega D_lst ali rezultatov.
Vse, kar ustvari, zapiše v mapo te skripte (results/, work/).

Podukazi (vsak teče v svojem procesu, da se pomnilnik sprosti):
  provenance   SHA-256 datotek, LFS kazalec iz gita, verzije paketov
  regen        ponovno ustvari celoten D_lst
                 --variant current      trenutna v7 + trenutni vhodi (v12)
                 --variant hist45d9509  koda iz commita 45d9509 (v5, 2025-07-25) + vhodi v09/v10
  digest       iz pickla izdela kompakten povzetek in preveri strukturo
                 --which existing | current | hist45d9509
  independent  neodvisna vektorska implementacija formule (brez erst)
  compare      primerja povzetke med seboj in z neodvisnim izračunom
  examples     sledljivi primeri računa od izvornih odgovorov do ocene
  all          vse zgoraj po vrsti

Primer:
  .venv\\Scripts\\python.exe -B diagnostics\\dlst_audit_20261027\\audit_dlst.py all
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
BASE_DIR = AUDIT_DIR.parents[1]
RESULTS = AUDIT_DIR / "results"
WORK = AUDIT_DIR / "work"

EXISTING_DLST = BASE_DIR / "data" / "D_lst" / "D_lst_full_20260202_150312.pkl"
EXISTING_COPY = BASE_DIR / "data" / "D_lst" / "D_lst_full.pkl"
V7 = BASE_DIR / "src" / "runGeneratorOfD_v7.py"
HIST_COMMIT = "45d9509"

PKL = {
    "existing": EXISTING_DLST,
    "current": WORK / "D_lst_regen_current.pkl",
    "hist45d9509": WORK / "D_lst_regen_hist45d9509.pkl",
}

ACTIVITY_QS = ["Ac_AB4_1", "Ac_AB4_2", "Ac_AB4_3", "Ac_AB4_4", "Ac_AB4_5", "Ac_AB4_8"]
R_T = 0.3
MAX_LEN = 3

EXPECTED_CALL_ARGS = [
    "uIDsIn", "seq_actID_lstIn", "uID_actID_answers_df", "actID_score_df",
    "actID_compat_df", "r_T", "meth_code",
    "actID_context_dc=actID_context_dc", "normalize=True",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=BASE_DIR, capture_output=True,
                          text=True, encoding="utf-8").stdout


# ======================================================================================
# 1. PROVENANCE
# ======================================================================================

def cmd_provenance(_):
    files = [
        EXISTING_DLST, EXISTING_COPY,
        BASE_DIR / "src/runGeneratorOfD_v7.py",
        BASE_DIR / "src/elderly_recsys_tools.py",
        BASE_DIR / "src/recommendation_items.py",
        BASE_DIR / "data/context/ActivityContextGen_v12.xlsx",
        BASE_DIR / "data/context/legacy/ActivityContextGen_v09.xlsx",
        BASE_DIR / "data/context/legacy/ActivityContextGen_v10.xlsx",
        BASE_DIR / "data/questionnaires/Activities.xlsx",
        BASE_DIR / "data/questionnaires/MentalHealth.xlsx",
        BASE_DIR / "data/questionnaires/PhysicalHealth.xlsx",
        BASE_DIR / "data/questionnaires/SocialHealth.xlsx",
        BASE_DIR / "data/scores/ml_data_scores_and_wgt.xlsx",
    ]
    out = {"files": {}}
    for p in files:
        st = p.stat()
        out["files"][str(p.relative_to(BASE_DIR))] = {
            "sha256": sha256(p), "size": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "git_blob": git("hash-object", str(p)).strip() if p.suffix != ".pkl" else None,
        }
    # LFS pointers for the D_lst that were committed to git
    out["git_lfs_pointers"] = {}
    for c in git("log", "--all", "--format=%h|%ad", "--date=iso", "--", "data/D_lst_full.pkl").split("\n"):
        if not c:
            continue
        h, d = c.split("|")
        out["git_lfs_pointers"][h] = {"date": d, "pointer": git("show", f"{h}:data/D_lst_full.pkl").strip()}
    # Input blobs in historical commit
    out["hist_commit_blobs"] = {}
    for line in git("ls-tree", HIST_COMMIT, "data/").splitlines():
        meta, name = line.split("\t", 1)
        out["hist_commit_blobs"][name] = meta.split()[2]
    out["run_record_sha256_D_lst"] = "9b8ae198de039fc07893c1231f3a55de74b7f0c2f7dd94416edd9401dd477203"
    out["existing_matches_run_record"] = out["files"][str(EXISTING_DLST.relative_to(BASE_DIR))]["sha256"] == out["run_record_sha256_D_lst"]
    import sklearn
    out["env"] = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                  "sklearn": sklearn.__version__, "platform": platform.platform()}
    dump_json(out, "provenance.json")
    print(json.dumps({k: v for k, v in out.items() if k != "hist_commit_blobs"}, indent=2, ensure_ascii=False))


# ======================================================================================
# 2. REGENERATION (exec of the generator up to STEP 11, then the exact get_dataMat call)
# ======================================================================================

def _prepare_hist_tree() -> Path:
    root = WORK / f"hist{HIST_COMMIT}"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    for src_name, dst in [("src/runGeneratorOfD_v5.py", "src/runGeneratorOfD_v5.py"),
                          ("src/elderly_recsys_tools.py", "src/elderly_recsys_tools.py")]:
        blob = subprocess.run(["git", "show", f"{HIST_COMMIT}:{src_name}"], cwd=BASE_DIR,
                              capture_output=True).stdout
        (root / dst).write_bytes(blob)
    # Inputs: current files that are byte-identical (same git blob) to those in the commit
    mapping = {
        "data/Activities.xlsx": "data/questionnaires/Activities.xlsx",
        "data/MentalHealth.xlsx": "data/questionnaires/MentalHealth.xlsx",
        "data/PhysicalHealth.xlsx": "data/questionnaires/PhysicalHealth.xlsx",
        "data/SocialHealth.xlsx": "data/questionnaires/SocialHealth.xlsx",
        "data/ml_data_scores_and_wgt.xlsx": "data/scores/ml_data_scores_and_wgt.xlsx",
        "data/ActivityContextGen_v09.xlsx": "data/context/legacy/ActivityContextGen_v09.xlsx",
        "data/ActivityContextGen_v10.xlsx": "data/context/legacy/ActivityContextGen_v10.xlsx",
    }
    tree = {}
    for line in git("ls-tree", HIST_COMMIT, "data/").splitlines():
        meta, name = line.split("\t", 1)
        tree[name] = meta.split()[2]
    for in_commit, local in mapping.items():
        local_blob = git("hash-object", str(BASE_DIR / local)).strip()
        if local_blob != tree[in_commit]:
            raise RuntimeError(f"{local} ni enak {in_commit} v {HIST_COMMIT}")
        shutil.copy2(BASE_DIR / local, root / in_commit)
    return root


def cmd_regen(args):
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.savefig = lambda *a, **k: None   # no figure files from the diagnostic run
    plt.show = lambda *a, **k: None

    WORK.mkdir(parents=True, exist_ok=True)
    if args.variant == "current":
        script = V7
        sys.path.insert(0, str(BASE_DIR / "src"))
        # STEP 1 of v7 calls mkdir(exist_ok=True) on output folders; require that they exist
        for d in ["evaluation", "latex/tabs", "latex/figs", "recommendations"]:
            if not (BASE_DIR / "outputs/heldout_evaluation" / d).is_dir():
                raise RuntimeError(f"Izhodna mapa {d} ne obstaja; v7 bi jo ustvaril. Prekinjam.")
    else:
        root = _prepare_hist_tree()
        script = root / "src" / "runGeneratorOfD_v5.py"
        sys.path.insert(0, str(root / "src"))

    src = script.read_text(encoding="utf-8-sig")
    cut = src.index("#%% STEP 11")
    prefix, rest = src[:cut], src[cut:]
    m = re.search(r"D_lst\s*=\s*erst\.get_dataMat\((.*?)\)\s*\n", rest, re.S)
    call_args = [a.strip() for a in m.group(1).split(",") if a.strip()]
    if [a.replace(" ", "") for a in call_args] != [a.replace(" ", "") for a in EXPECTED_CALL_ARGS]:
        raise RuntimeError(f"Klic get_dataMat se razlikuje od pričakovanega: {call_args}")

    ns = {"__file__": str(script), "__name__": "__dlst_audit__"}
    t0 = time.time()
    exec(compile(prefix, str(script), "exec"), ns)
    t_prep = time.time() - t0

    settings = {k: ns.get(k) for k in ["test_mode", "USE_EXISTING_D_LST", "act_max_len", "meth_code",
                                        "r_T", "aspect_groups_lst"]}
    if settings.get("test_mode"):
        raise RuntimeError("test_mode=True v generirni kodi; revizija zahteva celoten nabor.")
    info = {
        "variant": args.variant, "script": str(script), "script_sha256": sha256(script),
        "erst_file": ns["erst"].__file__, "erst_sha256": sha256(Path(ns["erst"].__file__)),
        "settings": settings,
        "n_users_in": len(ns["uIDsIn"]), "n_users_total": len(ns["uIDs"]),
        "n_single_actions": len(ns["actID_lstIn"]), "single_actions": list(ns["actID_lstIn"]),
        "n_combinations": len(ns["seq_actID_lstIn"]),
        "combinations_truncated": len(ns["seq_actID_lstIn"]) != len(ns["seq_actID_lst"]),
        "users_truncated": len(ns["uIDsIn"]) != len(ns["uIDs"]),
        "call_args": call_args, "prep_seconds": round(t_prep, 1),
    }
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))

    # Small intermediate objects for examples/diagnosis
    with open(WORK / f"intermediates_{args.variant}.pkl", "wb") as f:
        pickle.dump({
            "uIDsIn": ns["uIDsIn"], "actID_lstIn": ns["actID_lstIn"],
            "actID_qID_dc": ns["actID_qID_dc"], "actID_context_dc": ns["actID_context_dc"],
            "actID_score_df": ns["actID_score_df"], "uID_actID_answers_df": ns["uID_actID_answers_df"],
            "actID_compat_df": ns["actID_compat_df"],
            "a_F2": ns["uID_scores_dc"]["activity"],
        }, f)

    t0 = time.time()
    D_lst = eval("erst.get_dataMat(" + ", ".join(call_args) + ")", ns)
    info["generate_seconds"] = round(time.time() - t0, 1)
    info["rows"] = len(D_lst)
    out = PKL[args.variant]
    with open(out, "wb") as f:
        pickle.dump(D_lst, f)   # same call as v7 (default protocol)
    info["output"] = str(out)
    info["output_sha256"] = sha256(out)
    info["pickle_protocol_default"] = pickle.DEFAULT_PROTOCOL
    dump_json(info, f"regen_{args.variant}.json")
    print(f"rows={len(D_lst)}  sha256={info['output_sha256']}  {info['generate_seconds']} s")


# ======================================================================================
# 3. DIGEST + FULL STRUCTURAL CHECKS
# ======================================================================================

CTX_KEYS = ("qID", "C_T1", "C_T2", "C_T3", "C_P1", "C_P2", "C_P3")


def cmd_digest(args):
    path = PKL[args.which]
    t0 = time.time()
    with open(path, "rb") as f:
        D = pickle.load(f)
    load_s = time.time() - t0
    N = len(D)
    chk = {"which": args.which, "path": str(path), "sha256": sha256(path), "rows": N,
           "load_seconds": round(load_s, 1), "container_type": type(D).__name__}

    row_types, uid_types, item_types, ctx_types, rating_types = {}, {}, {}, {}, {}
    act_vocab, act_index = [], {}
    uid = np.empty(N, np.int64)
    acts = np.full((N, MAX_LEN), -1, np.int16)
    rating = np.empty(N, np.float64)
    lens = np.zeros(N, np.int8)
    bad_row_len = bad_item = repeat_within = 0
    ctx_len_mismatch = ctx_bad_keys = ctx_q_mismatch = ctx_differs_from_canonical = 0
    canonical_ctx = {}    # act -> repr(dict)
    ctx_variants = {}     # act -> set of reprs (should be one)
    examples = {"repeat_within": [], "ctx_differs": []}

    def inc(d, k):
        d[k] = d.get(k, 0) + 1

    for i, row in enumerate(D):
        inc(row_types, type(row).__name__)
        if len(row) != 4:
            bad_row_len += 1
            continue
        u, item, ctx, r = row
        inc(uid_types, type(u).__name__)
        inc(item_types, type(item).__name__)
        inc(ctx_types, type(ctx).__name__)
        inc(rating_types, type(r).__name__)
        uid[i] = int(u)
        rating[i] = float(r)
        if not isinstance(item, tuple) or not (1 <= len(item) <= MAX_LEN) or \
                any(not isinstance(a, str) for a in item):
            bad_item += 1
            continue
        lens[i] = len(item)
        if len(set(item)) != len(item):
            repeat_within += 1
            if len(examples["repeat_within"]) < 5:
                examples["repeat_within"].append(repr(row[:2]))
        for j, a in enumerate(item):
            k = act_index.get(a)
            if k is None:
                k = act_index[a] = len(act_vocab)
                act_vocab.append(a)
            acts[i, j] = k
        if not isinstance(ctx, tuple) or len(ctx) != len(item):
            ctx_len_mismatch += 1
            continue
        for a, c in zip(item, ctx):
            if not isinstance(c, dict) or tuple(c.keys()) != CTX_KEYS:
                ctx_bad_keys += 1
                continue
            rc = repr(c)
            ref = canonical_ctx.setdefault(a, rc)
            if rc != ref:
                ctx_differs_from_canonical += 1
                ctx_variants.setdefault(a, {ref}).add(rc)
                if len(examples["ctx_differs"]) < 5:
                    examples["ctx_differs"].append((i, a, rc, ref))
            if c["qID"] != a.rsplit("_Act", 1)[0]:
                ctx_q_mismatch += 1
    del D

    finite = np.isfinite(rating)
    chk.update({
        "row_types": row_types, "uid_types": uid_types, "item_types": item_types,
        "context_types": ctx_types, "rating_types": rating_types,
        "bad_row_len": bad_row_len, "bad_item": bad_item,
        "rows_with_repeated_actID_in_item": repeat_within,
        "context_len_mismatch": ctx_len_mismatch, "context_bad_keys": ctx_bad_keys,
        "context_qID_not_prefix_of_actID": ctx_q_mismatch,
        "context_dicts_differing_from_first_seen_for_same_actID": ctx_differs_from_canonical,
        "acts_with_multiple_context_variants": {a: sorted(v) for a, v in ctx_variants.items()},
        "rating_nan": int(np.isnan(rating).sum()), "rating_inf": int(np.isinf(rating).sum()),
        "rating_min": float(rating[finite].min()), "rating_max": float(rating[finite].max()),
        "rating_mean": float(rating[finite].mean()),
        "rating_below_rT": int((rating < R_T).sum()), "rating_above_5": int((rating > 5).sum()),
        "rating_eq_min_count": int((rating == rating[finite].min()).sum()),
        "rating_eq_5_count": int((rating == 5.0).sum()),
        "length_distribution": {int(k): int(v) for k, v in zip(*np.unique(lens, return_counts=True))},
        "n_users": int(len(np.unique(uid))), "n_actIDs": len(act_vocab),
        "examples": examples,
    })

    # duplicates: exact (uid, ordered item) and unordered (uid, sorted item)
    key_ordered = pd.DataFrame({"u": uid, "a0": acts[:, 0], "a1": acts[:, 1], "a2": acts[:, 2]})
    chk["duplicate_user_item_pairs"] = int(key_ordered.duplicated().sum())
    names = np.array(act_vocab + [""], dtype=object)
    srt = np.sort(np.where(acts < 0, len(act_vocab), acts), axis=1)  # pad sorts last
    key_unordered = pd.DataFrame({"u": uid, "a0": srt[:, 0], "a1": srt[:, 1], "a2": srt[:, 2]})
    chk["duplicate_user_item_pairs_ignoring_order"] = int(key_unordered.duplicated().sum())
    items_ordered = key_ordered[["a0", "a1", "a2"]].drop_duplicates()
    items_unordered = key_unordered[["a0", "a1", "a2"]].drop_duplicates()
    chk["n_distinct_items_ordered"] = len(items_ordered)
    chk["n_distinct_items_unordered"] = len(items_unordered)
    # order checks
    chk["uid_non_decreasing"] = bool(np.all(np.diff(uid) >= 0))
    chk["first_rows"] = [[int(uid[i]), [names[k] for k in acts[i] if k >= 0], float(rating[i])]
                         for i in range(min(3, N))]

    dg = {"uid": uid, "acts": acts, "rating": rating, "act_vocab": act_vocab,
          "canonical_ctx": canonical_ctx}
    with open(WORK / f"digest_{args.which}.pkl", "wb") as f:
        pickle.dump(dg, f, protocol=4)
    dump_json(chk, f"checks_{args.which}.json")
    print(json.dumps({k: v for k, v in chk.items() if k not in ("examples",)}, indent=2,
                     ensure_ascii=False, default=str))


# ======================================================================================
# 4. INDEPENDENT IMPLEMENTATION (does not import elderly_recsys_tools)
# ======================================================================================

def _action_table(path):
    a = pd.read_excel(path, sheet_name="ActionLst")
    a["qID_ff"] = a["qID"].ffill()
    a = a[a["actID"].notna()]
    act = a[a["qID_ff"].isin(ACTIVITY_QS)].copy()
    return act


def independent_scores(context_xlsx, compat_xlsx):
    """Formula as stated, computed with numpy from the raw Excel files.

    a_u   = answer of user u to the question q(a) of action a (raw, -1 -> NaN)
    s_u   = min-max scaled answer over all questionnaire rows (NaN -> 0 before scaling)
    v_ua  = w_u * a_u * s_u                (w_u = a_F2 weight)
    raw   = sum_{a in combo} v_ua - (1 - prod_{i<j} C[a_i, a_j])
    keep  raw > r_T; rating = r_T + (5 - r_T) * (raw - min) / (max - min) over kept rows
    """
    q = BASE_DIR / "data/questionnaires"
    ans = pd.read_excel(q / "Activities.xlsx", sheet_name="AllAnswers", index_col="S4").replace(-1, np.nan)
    ans.columns = ["Ac_" + c for c in ans.columns]
    w = pd.read_excel(BASE_DIR / "data/scores/ml_data_scores_and_wgt.xlsx", sheet_name="scores_and_wgt",
                      header=1, index_col="person_id")["a_F2"]
    users = np.sort(w.index.to_numpy())
    act = _action_table(context_xlsx)
    acts = act["actID"].astype(str).tolist()
    qs = act["qID_ff"].tolist()
    C = pd.read_excel(compat_xlsx, sheet_name="ActionCompatibilities", index_col=0).loc[acts, acts].to_numpy(float)

    X = ans[ACTIVITY_QS].astype(float)
    X0 = X.fillna(0.0)
    S = (X0 - X0.min()) / (X0.max() - X0.min())
    if ans.index.duplicated().any():
        assert not np.isin(ans.index[ans.index.duplicated()], users).any()
    Xu = X[~X.index.duplicated()].loc[users]
    Su = S[~S.index.duplicated()].loc[users]
    wu = w.loc[users].to_numpy(float)
    V = np.empty((len(users), len(acts)))
    for j, qq in enumerate(qs):
        V[:, j] = (wu * Xu[qq].to_numpy()) * Su[qq].to_numpy()

    combos = []
    for L in range(1, MAX_LEN + 1):
        combos += list(itertools.combinations(range(len(acts)), L))
    ncomb = len(combos)
    raw = np.empty((len(users), ncomb))
    for k, cb in enumerate(combos):
        s = np.zeros(len(users))
        for j in cb:
            s = s + V[:, j]
        comp = 1.0
        for i1, i2 in itertools.combinations(cb, 2):
            comp = comp * C[i1, i2]
        raw[:, k] = s - (1 - comp)
    keep = raw > R_T          # NaN -> False
    uu, kk = np.nonzero(keep)   # row-major: users ascending, then combo order
    r = raw[uu, kk]
    mn, mx = r.min(), r.max()
    rating = R_T + (5.0 - R_T) * (r - mn) / (mx - mn)
    return {"users": users, "acts": acts, "qs": qs, "combos": combos, "C": C, "V": V,
            "Xu": Xu, "Su": Su, "wu": wu, "raw": raw, "keep": keep, "uu": uu, "kk": kk,
            "rating": rating, "raw_min_kept": mn, "raw_max_kept": mx}


def cmd_independent(_):
    variants = {
        "indep_v12": (BASE_DIR / "data/context/ActivityContextGen_v12.xlsx",) * 2,
        "indep_v09_v10": (BASE_DIR / "data/context/legacy/ActivityContextGen_v09.xlsx",
                          BASE_DIR / "data/context/legacy/ActivityContextGen_v10.xlsx"),
    }
    summary = {}
    for name, (cx, cp) in variants.items():
        R = independent_scores(cx, cp)
        acts = R["acts"]
        combos = R["combos"]
        n = len(R["uu"])
        A = np.full((n, MAX_LEN), -1, np.int16)
        for j in range(MAX_LEN):
            A[:, j] = np.array([combos[k][j] if len(combos[k]) > j else -1 for k in range(len(combos))],
                               np.int16)[R["kk"]]
        dg = {"uid": R["users"][R["uu"]].astype(np.int64), "acts": A, "rating": R["rating"],
              "act_vocab": acts, "raw": R["raw"][R["uu"], R["kk"]]}
        with open(WORK / f"digest_{name}.pkl", "wb") as f:
            pickle.dump(dg, f, protocol=4)
        # Users with no kept rows, and why
        users = R["users"]
        kept_per_user = R["keep"].sum(1)
        missing = []
        for i in np.nonzero(kept_per_user == 0)[0]:
            x = R["Xu"].iloc[i]
            missing.append({
                "uid": int(users[i]), "a_F2": float(R["wu"][i]),
                "answers": {k: (None if pd.isna(v) else float(v)) for k, v in x.items()},
                "max_raw": None if np.all(np.isnan(R["raw"][i])) else float(np.nanmax(R["raw"][i])),
                "n_nan_raw": int(np.isnan(R["raw"][i]).sum()),
                "reason": ("vsi odgovori manjkajo (-1/NaN): vsi raw = NaN"
                           if x.isna().all() else
                           "vseh 6 odgovorov = 0: največji raw = 0 <= r_T"
                           if (x.fillna(-1) == 0).all() else
                           "mešano 0/NaN: največji raw <= r_T ali NaN"),
            })
        # Compat matrix properties on the used block
        C = R["C"]
        summary[name] = {
            "rows": n, "n_users_total": len(users), "n_users_included": int((kept_per_user > 0).sum()),
            "n_single_actions": len(acts), "n_combinations": len(combos),
            "raw_min_kept": float(R["raw_min_kept"]), "raw_max_kept": float(R["raw_max_kept"]),
            "raw_nan_pairs": int(np.isnan(R["raw"]).sum()),
            "raw_le_rT_pairs": int((R["raw"] <= R_T).sum()),
            "raw_in_(rT, rT+1e-9]": int(((R["raw"] > R_T) & (R["raw"] <= R_T + 1e-9)).sum()),
            "raw_exactly_rT": int((R["raw"] == R_T).sum()),
            "compat_nan": int(np.isnan(C).sum()), "compat_min": float(np.nanmin(C)),
            "compat_max": float(np.nanmax(C)), "compat_symmetric": bool(np.allclose(C, C.T, equal_nan=True)),
            "compat_distinct_values": sorted(set(np.round(C[np.triu_indices_from(C, 1)], 6).tolist())),
            "scaled_equals_raw_answer": bool(np.allclose(R["Su"].to_numpy(), R["Xu"].fillna(0).to_numpy())),
            "missing_users": missing,
        }
        print(name, {k: v for k, v in summary[name].items() if k != "missing_users"})
    dump_json(summary, "independent.json")


# ======================================================================================
# 5. COMPARE
# ======================================================================================

def _load_digest(name):
    with open(WORK / f"digest_{name}.pkl", "rb") as f:
        return pickle.load(f)


def _keys(dg):
    v = np.array(list(dg["act_vocab"]) + [""], dtype=object)
    A = np.where(dg["acts"] < 0, len(dg["act_vocab"]), dg["acts"])
    return dg["uid"], v[A[:, 0]], v[A[:, 1]], v[A[:, 2]]


def compare_pair(a_name, b_name):
    A, B = _load_digest(a_name), _load_digest(b_name)
    ua, a0, a1, a2 = _keys(A)
    ub, b0, b1, b2 = _keys(B)
    res = {"a": a_name, "b": b_name, "rows_a": len(ua), "rows_b": len(ub)}
    same_len = len(ua) == len(ub)
    if same_len:
        pos_same = (ua == ub) & (a0 == b0) & (a1 == b1) & (a2 == b2)
        res["positional_key_equal_rows"] = int(pos_same.sum())
        res["order_identical"] = bool(pos_same.all())
        if not pos_same.all():
            res["first_positional_mismatch_row"] = int(np.argmin(pos_same))
        ra, rb = A["rating"], B["rating"]
        bit_eq = ra.view(np.int64) == rb.view(np.int64)
        res["positional_rating_bitwise_equal"] = int((bit_eq & pos_same).sum())
    da = pd.DataFrame({"u": ua, "a0": a0, "a1": a1, "a2": a2, "ra": A["rating"]})
    db = pd.DataFrame({"u": ub, "a0": b0, "a1": b1, "a2": b2, "rb": B["rating"]})
    m = da.merge(db, on=["u", "a0", "a1", "a2"], how="outer", indicator=True)
    res["common_pairs"] = int((m["_merge"] == "both").sum())
    res["only_in_a"] = int((m["_merge"] == "left_only").sum())
    res["only_in_b"] = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    d = (both["ra"] - both["rb"]).abs()
    res["common_rating_bitwise_equal"] = int((both["ra"].to_numpy().view(np.int64) ==
                                              both["rb"].to_numpy().view(np.int64)).sum())
    res["common_rating_max_abs_diff"] = float(d.max()) if len(d) else None
    res["common_rating_n_diff_gt_1e-12"] = int((d > 1e-12).sum())
    res["examples_only_in_a"] = m[m["_merge"] == "left_only"].head(5)[["u", "a0", "a1", "a2", "ra"]].values.tolist()
    res["examples_only_in_b"] = m[m["_merge"] == "right_only"].head(5)[["u", "a0", "a1", "a2", "rb"]].values.tolist()
    if len(d) and d.max() > 0:
        res["examples_largest_diff"] = both.assign(d=d).nlargest(5, "d")[["u", "a0", "a1", "a2", "ra", "rb", "d"]].values.tolist()
    # unordered comparison as auxiliary diagnostic
    def unordered(df):
        s = np.sort(df[["a0", "a1", "a2"]].to_numpy(dtype=str), axis=1)
        return set(zip(df["u"].tolist(), map(tuple, s)))
    if res["only_in_a"] or res["only_in_b"]:
        sa, sb = unordered(da), unordered(db)
        res["unordered_only_in_a"] = len(sa - sb)
        res["unordered_only_in_b"] = len(sb - sa)
    users_a, users_b = set(np.unique(ua).tolist()), set(np.unique(ub).tolist())
    res["users_only_in_a"] = sorted(users_a - users_b)
    res["users_only_in_b"] = sorted(users_b - users_a)
    res["acts_only_in_a"] = sorted(set(A["act_vocab"]) - set(B["act_vocab"]))
    res["acts_only_in_b"] = sorted(set(B["act_vocab"]) - set(A["act_vocab"]))
    # context per actID (only for pickled digests)
    if "canonical_ctx" in A and "canonical_ctx" in B:
        ca, cb = A["canonical_ctx"], B["canonical_ctx"]
        res["context_differs_for_actIDs"] = {k: [ca[k], cb[k]] for k in sorted(set(ca) & set(cb)) if ca[k] != cb[k]}
    return res


def cmd_compare(_):
    available = [n for n in ["existing", "current", "hist45d9509", "indep_v12", "indep_v09_v10"]
                 if (WORK / f"digest_{n}.pkl").exists()]
    pairs = [("existing", "current"), ("existing", "hist45d9509"), ("existing", "indep_v09_v10"),
             ("existing", "indep_v12"), ("current", "indep_v12"), ("hist45d9509", "indep_v09_v10"),
             ("current", "hist45d9509")]
    out = {}
    for a, b in pairs:
        if a in available and b in available:
            print(f"== {a} vs {b}")
            out[f"{a}__vs__{b}"] = r = compare_pair(a, b)
            print(json.dumps({k: v for k, v in r.items() if not k.startswith("examples")}, indent=1,
                             ensure_ascii=False, default=str)[:3000])
    # byte-level pickle comparison
    out["pickle_sha256"] = {n: sha256(p) for n, p in PKL.items() if p.exists()}
    dump_json(out, "compare.json")


# ======================================================================================
# 6. WORKED EXAMPLES
# ======================================================================================

def cmd_examples(_):
    R = independent_scores(BASE_DIR / "data/context/ActivityContextGen_v12.xlsx",
                           BASE_DIR / "data/context/ActivityContextGen_v12.xlsx")
    E = _load_digest("existing")
    ue, e0, e1, e2 = _keys(E)
    ex_idx = pd.DataFrame({"u": ue, "a0": e0, "a1": e1, "a2": e2, "r": E["rating"]}).set_index(["u", "a0", "a1", "a2"])["r"]
    users, acts, combos, C = R["users"], R["acts"], R["combos"], R["C"]
    uidx = {u: i for i, u in enumerate(users)}
    mn, mx = R["raw_min_kept"], R["raw_max_kept"]

    def explain(u, cb, label):
        i = uidx[u]
        k = combos.index(cb)
        names = [acts[j] for j in cb]
        parts = []
        for j in cb:
            q = R["qs"][j]
            parts.append({"actID": acts[j], "qID": q, "answer_raw": None if pd.isna(R["Xu"].iloc[i][q]) else float(R["Xu"].iloc[i][q]),
                          "answer_scaled": float(R["Su"].iloc[i][q]), "w_a_F2": float(R["wu"][i]),
                          "contribution": float(R["V"][i, j])})
        pairs = [{"pair": [acts[a], acts[b]], "C": float(C[a, b])} for a, b in itertools.combinations(cb, 2)]
        comp = float(np.prod([p["C"] for p in pairs])) if pairs else 1.0
        raw = float(R["raw"][i, k])
        kept = bool(raw > R_T)
        norm = R_T + (5 - R_T) * (raw - mn) / (mx - mn) if kept else None
        key = (int(u), names[0], names[1] if len(names) > 1 else "", names[2] if len(names) > 2 else "")
        existing = float(ex_idx.get(key)) if key in ex_idx.index else None
        return {"label": label, "uid": int(u), "item": names, "parts": parts, "pairs": pairs,
                "sum_contrib": float(sum(p["contribution"] for p in parts)), "prod_compat": comp,
                "penalty_1_minus_prod": 1 - comp, "raw": raw, "kept_raw_gt_rT": kept,
                "normalized": norm, "normalization": {"min_raw": float(mn), "max_raw": float(mx)},
                "existing_D_lst_rating": existing,
                "match_existing_bitwise": bool(existing is not None and norm is not None and
                                               np.float64(existing).view(np.int64) == np.float64(norm).view(np.int64))}

    ex = []
    u = 1373 if 1373 in uidx else users[0]
    i = uidx[u]
    # one / two / three actions that are kept
    for L in (1, 2, 3):
        for k, cb in enumerate(combos):
            if len(cb) == L and R["keep"][i, k]:
                ex.append(explain(u, cb, f"{L} aktivnost(i), vključeno"))
                break
    # compatible (prod C == 1) vs incompatible (prod C < 1) pair for the same user
    done_c = done_i = False
    for k, cb in enumerate(combos):
        if len(cb) != 2:
            continue
        c = C[cb[0], cb[1]]
        if not done_c and c == 1 and R["keep"][i, k] and cb not in [tuple(e) for e in []]:
            ex.append(explain(u, cb, "par, združljiv (C=1)")); done_c = True
        if not done_i and c < 1 and np.isfinite(R["raw"][i, k]):
            ex.append(explain(u, cb, f"par, nezdružljiv (C={c})")); done_i = True
        if done_c and done_i:
            break
    # threshold: smallest kept raw and largest excluded raw (globally)
    raw = R["raw"]
    kept_vals = np.where(R["keep"], raw, np.inf)
    ii, kk = np.unravel_index(np.argmin(kept_vals), raw.shape)
    ex.append(explain(users[ii], combos[kk], "prag: najmanjši vključeni raw"))
    excl = np.where((~R["keep"]) & np.isfinite(raw), raw, -np.inf)
    ii, kk = np.unravel_index(np.argmax(excl), raw.shape)
    ex.append(explain(users[ii], combos[kk], "prag: največji izključeni raw"))
    # the maximum (rating 5)
    ii, kk = np.unravel_index(np.nanargmax(np.where(R["keep"], raw, -np.inf)), raw.shape)
    ex.append(explain(users[ii], combos[kk], "največji raw (ocena 5)"))
    # distinct raw values that survive, for interpretation
    vals, cnt = np.unique(np.round(raw[R["keep"]], 9), return_counts=True)
    dump_json({"examples": ex, "distinct_kept_raw_values": dict(zip(map(float, vals), map(int, cnt)))},
              "examples.json")
    for e in ex:
        print(e["label"], e["uid"], e["item"], "raw", e["raw"], "norm", e["normalized"], "existing",
              e["existing_D_lst_rating"], "bit-eq", e["match_existing_bitwise"])


# ======================================================================================

def cmd_all(_):
    me = [sys.executable, "-B", str(Path(__file__).resolve())]
    steps = [["provenance"], ["digest", "--which", "existing"],
             ["regen", "--variant", "current"], ["digest", "--which", "current"],
             ["regen", "--variant", "hist45d9509"], ["digest", "--which", "hist45d9509"],
             ["independent"], ["compare"], ["examples"]]
    for s in steps:
        print("\n>>>", " ".join(s), flush=True)
        subprocess.run(me + s, check=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("provenance")
    p = sp.add_parser("regen"); p.add_argument("--variant", choices=["current", "hist45d9509"], required=True)
    p = sp.add_parser("digest"); p.add_argument("--which", choices=list(PKL), required=True)
    sp.add_parser("independent"); sp.add_parser("compare"); sp.add_parser("examples"); sp.add_parser("all")
    a = ap.parse_args()
    {"provenance": cmd_provenance, "regen": cmd_regen, "digest": cmd_digest, "independent": cmd_independent,
     "compare": cmd_compare, "examples": cmd_examples, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
