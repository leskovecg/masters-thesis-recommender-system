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
| `samples/*.csv` | Example rows of both datasets (first/last 50, 500 random, 3 whole users, matrix blocks). They contain rows of individual users and are therefore **not in git**. |

Dataset labels: `D_lst_full_20260202_150312` stands for both D_lst_full files, since they are byte-identical. `D_contx_a3_sparse_mat` is the mentor-supplied file.

CSV conventions: UTF-8 with `\n` line endings. `rating` / `value` are written with Python `repr`, which round-trips exactly. `item` is a JSON list of actIDs (or `actID|actID|…` in matrix files). In context columns, an **empty cell is float NaN**, while the text `nan` is the literal string `'nan'`.

## How it was produced

With the script `summarize_dlst.py` (v1.0), run on Windows with the local .venv (Python 3.11.9) on the three pickles above.
**The script is not in the repository or its git history**, so the content digests below cannot be recomputed with it;
the raw SHA-256 can be checked with any tool. The segment tables (M3/M4/M5) use `data/context/ActivityContextGen_v12.xlsx`.
`README.md` and `provenance.md` are written by hand.

## Verifying a regenerated D_lst

1. Regenerate: set `USE_EXISTING_D_LST = False` in `src/runGeneratorOfD_v7.py` and run it up to STEP 6 (it writes `data/D_lst/D_lst_full_<timestamp>.pkl`).
   The audit `diagnostics/dlst_audit_20261027/audit_dlst.py regen --variant current` does the same with the code up to commit `0df8fdd`.
2. Compare the raw SHA-256 with the one in the table above:

   ```bash
   sha256sum data/D_lst/D_lst_full_<timestamp>.pkl
   ```

   An equal hash means an identical file. A different hash may still be the same content in another pickle protocol or library version.

How to read the result:

| Digest in `fingerprints.json` | Meaning if it matches |
|---|---|
| `sha256` (raw file) | Byte-identical pickle. Also requires the same pickle protocol and the same numpy pickling format. |
| `fingerprint.content_sha256` | **Identical content**, including row order, every rating bit and every context value. It does not depend on pickle protocol, library versions or dict key order. This is the criterion for "identical D_lst". |
| `fingerprint.content_sha256_unordered` | Same multiset of rows; only the order may differ. Row order matters for v7's `ShuffleSplit(random_state=42)`, so this alone is **not** sufficient. |
| `fingerprint.triplets_sha256` | Same `(user_id, item, rating)` sequence, which is everything Surprise sees. Contexts may differ. |
| `fingerprint.component_sha256.{user_id,item,context,rating}` | Pinpoints which column differs. |

The digest was computed by `canon()` in `summarize_dlst.py` (not in the repository). It encodes the loaded object as type-tagged text: ints and floats by value (floats as exact `float.hex()`), strings with their length, dict keys sorted, and list and tuple kept distinct. `content_sha256` is the SHA-256 of that text. It is value-based: `numpy.int64(2)` and `int 2` hash the same. Exact Python types are recorded in `schema.md`. A string `'nan'` and a float NaN hash differently.

If the raw SHA-256 differs but `content_sha256` matches, the files differ only in serialisation (e.g. pickle protocol or numpy version) and the content is the same.
