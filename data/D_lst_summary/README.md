# D_lst summary (text-only stand-in for the git-ignored pickles)

The rating datasets `data/D_lst/*.pkl` (~555 MB each) are not in git (`*.pkl` is in `.gitignore`).
This folder describes them in `.md` / `.json` / `.csv` files so that a reviewer can:

1. understand what they contain without opening the pickles, and
2. check that a regenerated D_lst is **identical** to the original.

| Summarised file | Size (bytes) | SHA-256 (raw file) |
|---|---:|---|
| `data/D_lst/D_lst_full_20260202_150312.pkl` | 581,511,163 | `9b8ae198de039fc07893c1231f3a55de74b7f0c2f7dd94416edd9401dd477203` |
| `data/D_lst/D_lst_full.pkl` | 581,511,163 | identical (byte-for-byte copy) |
| `data/D_lst/D_contx_a3_sparse_mat.pkl` | 59,154,473 | `b28b40e35594b259de544e1fc0b94be7772bdd7f1e4c2192d7af166991620c27` |

## Files

| File | Content |
|---|---|
| [provenance.md](provenance.md) | Which code writes and reads each pickle (file:line), what one D_lst element means, how it is built (inputs, parameters, seeds, ActivityContextGen versions), and whether it can be regenerated. Written by hand. |
| [fingerprints.json](fingerprints.json) | Raw SHA-256 and size of every file, pickle protocol, canonical content digests (see below), pairwise comparisons, and the Python / numpy / pandas / scipy / scikit-surprise versions used. |
| [schema.md](schema.md) | Recursive structure of each dataset (types, lengths, dict keys and value summaries at every nesting level, collected over **all** rows), plus fully expanded example elements. |
| [stats.json](stats.json) / [stats.md](stats.md) | Counts (users, items, activities, context segments, ratings, sparsity), rating distribution (quantiles, NaN/inf, fixed-bin histogram), per-user / per-item / per-activity / per-segment coverage, user × item matrix view (shape, nnz, density, row/column nnz), and flags for anything suspicious. |
| `samples/<dataset>__first50.csv`, `__last50.csv` | First and last 50 rows, with flattened contexts (`ctx{1..3}_{qID,C_T1..C_P3}`). |
| `samples/<dataset>__random500_seed42.csv` | 500 random rows (`numpy.random.default_rng(42)`, without replacement, sorted by row index). |
| `samples/<dataset>__users3_seed42_all_rows.csv` | Every row of 3 users drawn with `default_rng(42)` from the sorted user list. Contexts are omitted to save space; the column `context_equals_activity_contexts_csv` confirms that each row's context equals the per-activity table. |
| `samples/<dataset>__activity_contexts.csv` | Context dict of each of the 34 actIDs, which is the same in every row containing it. |
| `samples/<dataset>__matrix_dense_block_50x50.csv` | 50 × 50 block of the user × item matrix: the 50 most-filled items, then the 50 users with most entries among them. An empty cell means no rating. |
| `samples/<dataset>__matrix_random200_nonzero_seed42.csv` | 200 random stored (user, item, value) entries (`default_rng(42)`). |

Dataset labels: `D_lst_full_20260202_150312` stands for both D_lst_full files, since they are byte-identical. `D_contx_a3_sparse_mat` is the mentor-supplied file.

CSV conventions: UTF-8 with `\n` line endings. `rating` / `value` are written with Python `repr`, which round-trips exactly. `item` is a JSON list of actIDs (or `actID|actID|…` in matrix files). In context columns, an **empty cell is float NaN**, while the text `nan` is the literal string `'nan'`.

## How it was produced

```powershell
# from the repo root, Windows, local .venv (Python 3.11.9)
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -B src\tools\summarize_dlst.py `
    --input data\D_lst\D_lst_full_20260202_150312.pkl data\D_lst\D_lst_full.pkl data\D_lst\D_contx_a3_sparse_mat.pkl `
    --output-dir data\D_lst_summary
```

Runtime is about 10-15 minutes and peak RAM about 10 GB, because one D_lst is held in memory at a time. A file that is byte-identical to an earlier input is not loaded again. The segment tables (M3/M4/M5) use `data/context/ActivityContextGen_v12.xlsx` (option `--context-xlsx`). `README.md` and `provenance.md` are written by hand; the script does not overwrite them.

## Verifying a regenerated D_lst

1. Regenerate. Either set `USE_EXISTING_D_LST = False` in `src/runGeneratorOfD_v7.py` and run it up to STEP 11 (it writes `data/D_lst/D_lst_full_<timestamp>.pkl`), or run `diagnostics/dlst_audit_20261027/audit_dlst.py regen_current`.
2. Compare with the committed fingerprints:

   ```powershell
   .\.venv\Scripts\python.exe -B src\tools\summarize_dlst.py --fingerprint-only `
       --input data\D_lst\D_lst_full_<timestamp>.pkl --verify data\D_lst_summary\fingerprints.json
   ```

   This prints `MATCH` / `differs` for each digest against the closest reference and exits with code 0 only if `content_sha256` matches.

How to read the result:

| Digest in `fingerprints.json` | Meaning if it matches |
|---|---|
| `sha256` (raw file) | Byte-identical pickle. Also requires the same pickle protocol and the same numpy pickling format. |
| `fingerprint.content_sha256` | **Identical content**, including row order, every rating bit and every context value. It does not depend on pickle protocol, library versions or dict key order. This is the criterion for "identical D_lst". |
| `fingerprint.content_sha256_unordered` | Same multiset of rows; only the order may differ. Row order matters for v7's `ShuffleSplit(random_state=42)`, so this alone is **not** sufficient. |
| `fingerprint.triplets_sha256` | Same `(user_id, item, rating)` sequence, which is everything Surprise sees. Contexts may differ. |
| `fingerprint.component_sha256.{user_id,item,context,rating}` | Pinpoints which column differs. |

The digest is computed by `canon()` in [src/tools/summarize_dlst.py](../../src/tools/summarize_dlst.py), which is documented in the module docstring. It encodes the loaded object as type-tagged text: ints and floats by value (floats as exact `float.hex()`), strings with their length, dict keys sorted, and list and tuple kept distinct. `content_sha256` is the SHA-256 of that text. It is value-based: `numpy.int64(2)` and `int 2` hash the same. Exact Python types are recorded in `schema.md`. A string `'nan'` and a float NaN hash differently.

If the raw SHA-256 differs but `content_sha256` matches, the files differ only in serialisation (e.g. pickle protocol or numpy version) and the content is the same.
