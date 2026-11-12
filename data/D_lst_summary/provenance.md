# Provenance of the D_lst pickles

Covers the three files summarised in this folder:

| File | Size (bytes) | Role |
|---|---:|---|
| `data/D_lst/D_lst_full_20260202_150312.pkl` | 581,511,163 | D_lst used by the final run `20261015_150006` |
| `data/D_lst/D_lst_full.pkl` | 581,511,163 | older name of the same content (byte-identical copy) |
| `data/D_lst/D_contx_a3_sparse_mat.pkl` | 59,154,473 | mentor-supplied D_lst from an earlier stage (comparison only) |

Line numbers refer to commit `355a6d2` unless a different commit is given.

## 1. Where the files are written and read

### Written

| Location | What |
|---|---|
| [src/runGeneratorOfD_v7.py:478-482](../../src/runGeneratorOfD_v7.py#L478-L482) | `pickle.dump(D_lst, f)` writes `data/D_lst/D_lst_full_<YYYYMMDD_HHMMSS>.pkl` (default pickle protocol). This branch only runs when `USE_EXISTING_D_LST = False`. The flag is currently `True` ([:88](../../src/runGeneratorOfD_v7.py#L88)). |
| [src/legacy/runGeneratorOfD_v6.py:474](../../src/legacy/runGeneratorOfD_v6.py#L474), [src/legacy/runGeneratorOfD_v5.py:470](../../src/legacy/runGeneratorOfD_v5.py#L470) | Same writer in older versions, also with timestamped names. |
| `45d9509:src/runGeneratorOfD_v5.py:439-440` (git history, 2025-07-25) | The only code that ever wrote the fixed name `data/D_lst_full.pkl`. This is the origin of the name `D_lst_full.pkl`. |
| *(none)* | **No code in the repository or in its git history writes `D_contx_a3_sparse_mat.pkl`.** |

### Read

| Location | What |
|---|---|
| [src/runGeneratorOfD_v7.py:112](../../src/runGeneratorOfD_v7.py#L112), [:444-445](../../src/runGeneratorOfD_v7.py#L444-L445) | `EXISTING_D_LST_PATH = data/D_lst/D_lst_full_20260202_150312.pkl` is loaded with `pickle.load`. This is the production path. |
| [notebooks/Dlst_analysis_v2.ipynb](../../notebooks/Dlst_analysis_v2.ipynb) (JSON lines 30, 680-686) | `load_dlst()` loads `D_lst_full_20260202_150312.pkl`. Note: the path is a hard-coded absolute path on another machine. |
| [notebooks/Dlst_analysis.ipynb](../../notebooks/Dlst_analysis.ipynb) (JSON lines 117-126) | Loads both `D_lst_full.pkl` ("My D_lst") and `D_contx_a3_sparse_mat.pkl` ("Mentor's D_lst") and compares them. |
| [src/legacy/runGeneratorOfD_v5.py:474](../../src/legacy/runGeneratorOfD_v5.py#L474) | A commented-out load of `D_contx_a3_sparse_mat.pkl`, annotated *"Mentor's new and already built D_lst matrix"*. |
| `2953037:src/runGeneratorOfD_v5.py:290-296` (2025-06-19) | An active load of the mentor file. At that time the pipeline trained on the mentor's D_lst instead of generating its own. |
| `notebooks/D_contx_a3_sparse_mat.ipynb` (added `050dcb8` 2025-06-14, deleted `94825c4` 2025-07-26) | A small inspection notebook that loaded and described the mentor file (first/last element, basic stats). |

`diagnostics/dlst_audit_20261027/audit_dlst.py` also reads and regenerates D_lst. It is out of scope here, but its findings are summarised in section 2.

## 2. What one element of D_lst is (`D_lst_full*.pkl`)

`D_lst` is a Python `list` with 3,053,029 rows. Each row is a `list` of length 4:

```
[ user_id, item, context, rating ]
  numpy.int64   tuple[str, ...]   tuple[dict, ...]   numpy.float64
```

* **user_id**: `person_id` of a respondent from `data/scores/ml_data_scores_and_wgt.xlsx`.
* **item**: a combination of 1-3 activity IDs (`actID`, e.g. `'Ac_AB4_1_Act01'`). It comes from `itertools.combinations` over the 34 activities linked to questions `Ac_AB4_{1,2,3,4,5,8}`, so there are no permutations and no repeats: 34 + 561 + 5,984 = 6,579 items ([elderly_recsys_tools.py:37-54](../../src/elderly_recsys_tools.py#L37-L54)).
* **context**: one dict per activity in the item, in the same order, with the keys `qID, C_T1, C_T2, C_T3, C_P1, C_P2, C_P3`. These are time and place conditions copied from the `act_C_T*` / `act_C_P*` columns of the `ActionLst` sheet. Missing values are **float NaN**. The context does not affect the rating.
* **rating**: relevance of the item for the user, on the scale [0.3, 5].

### How a row is built ([runGeneratorOfD_v7.py](../../src/runGeneratorOfD_v7.py) STEP 2-11, [elderly_recsys_tools.get_dataMat](../../src/elderly_recsys_tools.py#L56-L120))

1. **Inputs**
   - `data/questionnaires/Activities.xlsx` (`AllAnswers`, index `S4`, `-1 → NaN`): binary answers `AB4_1..5, AB4_8`.
   - `data/scores/ml_data_scores_and_wgt.xlsx` (`scores_and_wgt`): user list (`person_id`) and weight `a_F2`.
   - `data/context/ActivityContextGen_v12.xlsx`: sheet `ActionLst` (actID ↔ qID, contexts, row order) and sheet `ActionCompatibilities` (34 × 34 binary pairwise compatibility `C`).
   - Mental, physical and social health questionnaires are joined but do not influence D_lst (aspect = `activity` only).
2. **Per-activity contribution**: `v[u,a] = a_F2[u] · answer[u,q(a)] · scaled_answer[u,q(a)]`. For binary answers this equals `a_F2 · answer`.
3. **Raw score of an item**: `raw = Σ v − (1 − Π C[a_i,a_j])`. The penalty is 0, or exactly 1 if any pair in the item is incompatible.
4. **Threshold**: a row is kept only if `raw > r_T = 0.3`. Of the 696 users, 679 keep at least one row. The other 17 answered 0 to all six questions.
5. **Normalisation**: all kept rows are min-max scaled to `[r_T, 5] = [0.3, 5]`. The min/max are taken over the whole D_lst (raw min 0.388527, raw max 15.0).
6. **Order**: rows are grouped by user (users sorted ascending). Within a user, items follow the `itertools.combinations` order (length 1, then 2, then 3). This order matters: v7 STEP 12 splits with `ShuffleSplit(random_state=42)` on row positions.

**Parameters** ([runGeneratorOfD_v7.py:95-97](../../src/runGeneratorOfD_v7.py#L95-L97)): `act_max_len = 3`, `meth_code = 'score'`, `r_T = 0.3`, `normalize=True`, `aspect_groups_lst = ['activity']`.

**Random seeds**: `np.random.seed(42)` and `random.seed(42)` ([:61-62](../../src/runGeneratorOfD_v7.py#L61-L62)). D_lst generation itself is **deterministic** and uses no randomness.

**ActivityContextGen versions**: v7 reads **v12**. The content of `D_lst_full*.pkl` already existed on 2025-07-25 (git LFS pointer `45d9509:data/D_lst_full.pkl`, oid `sha256:9b8ae198…`), when the inputs were **v09/v10**. The audit (`diagnostics/dlst_audit_20261027/REPORT.md`) regenerated D_lst with the current v7 + v12 and with the historical v5 + v09/v10. Both produced a byte-identical pickle (same SHA-256). For these 34 activities, v12 and v09/v10 give the same actIDs, order, contexts and compatibilities. The date `20260202_150312` in the filename is **not** the creation date of the content.

**Regenerable**: yes. Set `USE_EXISTING_D_LST = False` in v7 and run STEP 0-11, or use `diagnostics/dlst_audit_20261027/audit_dlst.py regen_current`. Then compare the result with `fingerprints.json` (see [README.md](README.md)).

## 3. `D_contx_a3_sparse_mat.pkl`

**Despite its name, it is not a scipy sparse matrix.** It is a D_lst in the same `[user_id, item, context, rating]` list format (1,850,178 rows, 679 users, 6,579 items, 34 actIDs, ratings 0.31-5.0). See `schema.md` and `stats.md` for the exact structure. The name refers to its role as the sparse user × item rating matrix used for matrix factorisation ("contx" = with context, "a3" = up to 3 activities).

**Who created it**: the thesis mentor. Code and notebooks consistently call it *"Mentor's new and already built D_lst matrix"* (`2953037:src/runGeneratorOfD_v5.py:290`, [src/legacy/runGeneratorOfD_v5.py:474](../../src/legacy/runGeneratorOfD_v5.py#L474)) and *"Mentor's D_lst"* ([notebooks/Dlst_analysis.ipynb](../../notebooks/Dlst_analysis.ipynb)). It first appears in the repository on 2025-06-14 (`4bbfc37`, Git LFS, oid `sha256:b28b40e3…`, 59,154,473 bytes). It was removed from git tracking in `747d352` (2026-02-12) and is now only present locally.

**From what**: going by its structure, the same kind of generator applied to the same 34 activities and 679 users, but with a different (earlier) rule set. Differences visible in the data (details in `stats.md`):
- Missing context values are the **string `'nan'`**, not float NaN. This suggests the context columns were converted with `str()`, which current code never does.
- The context vocabulary differs from v12, e.g. `'ob zajtrkui'`, `'ob malici'` (v12: `'ob zajtrku'`, `'malica dopoldne'`).
- It has fewer rows (1.85 M vs 3.05 M) and a different rating distribution (min 0.31, mean 1.289 vs 0.30 / 1.541). So the scoring, threshold and/or normalisation rules differ.

**Can it be regenerated by existing code? No.** No code that creates it exists in the repository or in its git history. Every reference only *loads* it. The generating script, its parameters and its exact input versions are unknown. Its content can only be verified against the fingerprints in this folder (`fingerprints.json`), not reproduced. The current pipeline (v7) does not use it.
