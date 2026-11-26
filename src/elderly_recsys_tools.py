"""
elderly_recsys_tools.py
=======================

Tools for the master's thesis "Kontekstualiziran priporočilni sistem za dnevne aktivnosti
starejših" (used by runGeneratorOfD_v7.py).

Key features:
- Build D_lst (list of [user, activity combination, context, rating])
- Activity combinations: identity, context feasibility, group membership
- Simulated user contexts
- Cross-validation (ShuffleSplit is default for thesis)
- Top-k evaluation (Precision/Recall/F1, strict and tie-aware)
- Export tables to LaTeX and plot the results overview
"""
from __future__ import annotations

import itertools
import json
import numbers
import random
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from surprise import accuracy
from surprise.model_selection import KFold, ShuffleSplit
from tqdm import tqdm


# ======================================================================================
# DATA GENERATION
# ======================================================================================

def get_list_of_actions(single_act_lst: list[str], act_max_len: int) -> list[tuple[str]]:
    """
    Generates all possible combinations of actions up to a given length.

    Parameters:
    - single_act_lst (list): List of individual actions
    - act_max_len (int): Maximum length of action sequences to generate

    Returns:
    - seq_act_lst (list): List of action sequences (as tuples)
    """

    seq_act_lst = []

    for act_len in range(1, act_max_len+1):
        seq_act_lst += list(itertools.combinations(single_act_lst, act_len))

    return seq_act_lst

def get_dataMat(uIDs, seq_act_lst, uID_actID_answers_df, actID_score_df, compat_df, r_T, meth_code, actID_context_dc=None, normalize=True):
    """
    Builds a list of user-action sequence triples with associated contexts and scores.

    Parameters:
    - uIDs (list): List of user IDs
    - seq_act_lst (list): List of action sequences
    - uID_actID_answers_df (DataFrame): User answers for action questions
    - actID_score_df (DataFrame): Precomputed action scores per user
    - compat_df (DataFrame): Compatibility scores between actions
    - r_T (float): Threshold to include a recommendation
    - meth_code (str): Method to compute the rating
    - actID_context_dc (dict): Dictionary with full context for each action ID
    - normalize (bool): If True, rescale the kept ratings linearly to [r_T, 5]

    Returns:
    - D_lst (list): List of [user_id, action_sequence, context_sequence, score]
    """
    D_lst = []
    total = len(uIDs) * len(seq_act_lst)

    with tqdm(total=total, desc="Building D_lst") as pbar:
        for uID in uIDs:
            for act_seq in seq_act_lst:
                c_r = get_score_estimation(
                    uID, act_seq, uID_actID_answers_df, actID_score_df, compat_df, meth_code
                )

                if c_r > r_T:
                    if actID_context_dc is not None:
                        # context of every activity in the combination
                        context_sequence = tuple([
                            {
                                'qID': actID_context_dc[act]['qID'],
                                'C_T1': actID_context_dc[act]['act_C_T1'],
                                'C_T2': actID_context_dc[act]['act_C_T2'],
                                'C_T3': actID_context_dc[act]['act_C_T3'],
                                'C_P1': actID_context_dc[act]['act_C_P1'],
                                'C_P2': actID_context_dc[act]['act_C_P2'],
                                'C_P3': actID_context_dc[act]['act_C_P3']
                            }
                            for act in act_seq
                        ])
                    else:
                        context_sequence = ()

                    D_lst.append([uID, act_seq, context_sequence, c_r])
                
                pbar.update(1)

        # Normalization
        if normalize and D_lst:
            scores = [x[3] for x in D_lst]
            min_r, max_r = min(scores), max(scores)

            min_target = r_T
            max_target = 5.0

            if max_r > min_r:
                for entry in D_lst:
                    entry[3] = min_target + (max_target - min_target) * (entry[3] - min_r) / (max_r - min_r)
            else:
                for entry in D_lst:
                    entry[3] = (min_target + max_target) / 2  # all ratings are equal

    return D_lst

# ======================================================================================
# CONTEXT
# ======================================================================================

def is_action_context_feasibleQ(actID: str, cntx: dict, actID_context_dc: dict) -> bool:
    """
    Checks if actID is feasible in a given context.

    cntx expected keys:
      - 'C_T' (time label)
      - 'C_P' (place label)

    actID_context_dc[actID] contains:
      - act_C_T1..3, act_C_P1..3

    Labels are compared after stripping. "kadarkoli" (any time) and "kjerkoli" (anywhere)
    remove the restriction, in the activity definition as well as in the selected context.
    An unknown actID is not feasible.
    """
    if actID not in actID_context_dc:
        return False

    f_cntx = actID_context_dc[actID]
    C_Ts = [f_cntx[k].strip() for k in ['act_C_T1', 'act_C_T2', 'act_C_T3']
            if isinstance(f_cntx.get(k, None), str)]
    C_Ps = [f_cntx[k].strip() for k in ['act_C_P1', 'act_C_P2', 'act_C_P3']
            if isinstance(f_cntx.get(k, None), str)]

    user_C_T = (cntx.get('C_T', '') or '').strip()
    user_C_P = (cntx.get('C_P', '') or '').strip()

    time_ok = bool(user_C_T) and (user_C_T == 'kadarkoli' or 'kadarkoli' in C_Ts or user_C_T in C_Ts)
    place_ok = bool(user_C_P) and (user_C_P == 'kjerkoli' or 'kjerkoli' in C_Ps or user_C_P in C_Ps)
    return time_ok and place_ok


def build_context_pool(actID_context_dc: dict):
    """Returns the sorted time and place labels that occur in the given activity contexts."""
    C_T_pool, C_P_pool = set(), set()
    for _, f_cntx in actID_context_dc.items():
        for k in ["act_C_T1", "act_C_T2", "act_C_T3"]:
            v = f_cntx.get(k, None)
            if isinstance(v, str) and v.strip():
                C_T_pool.add(v.strip())
        for k in ["act_C_P1", "act_C_P2", "act_C_P3"]:
            v = f_cntx.get(k, None)
            if isinstance(v, str) and v.strip():
                C_P_pool.add(v.strip())
    return sorted(list(C_T_pool)), sorted(list(C_P_pool))


def sample_random_context(C_T_pool, C_P_pool):
    """Draws one simulated context (time, place); uses the global `random` state. Pools must be nonempty."""
    return {"C_T": random.choice(C_T_pool), "C_P": random.choice(C_P_pool)}


# ======================================================================================
# ITEMS (activity combinations)
# ======================================================================================
# An item is a nonempty tuple of activity IDs. Tuple order is retained, because
# the Surprise model uses that exact tuple as its key.

def normalize_item(item):
    if isinstance(item, str):
        item = (item,)
    elif isinstance(item, (tuple, list)):
        item = tuple(item)
    else:
        raise TypeError("Expected an activity ID or a tuple/list of activity IDs")
    if not item or any(not isinstance(a, str) or not a.strip() for a in item):
        raise ValueError("An item must contain nonempty string activity IDs")
    return item


def serialize_item(item):
    """Lossless JSON representation for Excel exports (not a model key)."""
    return json.dumps(normalize_item(item), ensure_ascii=False)


def item_context_feasible(item, context, actID_context_dc):
    """
    Every member must pass the SAME context check.

    This is a shared-context bundle, not a multi-context day schedule.
    """
    return all(is_action_context_feasibleQ(a, context, actID_context_dc)
               for a in normalize_item(item))


def item_in_group(item, group, mapping):
    """
    An item belongs to a group only if every member does.

    Each actID must be present in the mapping and its group must equal the
    requested group. Mixed-group combinations and items with an unmapped
    actID belong to no group; they are never assigned by first member.
    The same rule is applied to candidates and to the grouped reference.
    """
    return all(a in mapping and int(mapping[a]) == int(group)
               for a in normalize_item(item))


def first_k(ranked, k, keep):
    """First k recommendations (uid, item, score) of a ranked list whose item satisfies keep(item)."""
    return list(itertools.islice((rec for rec in ranked if keep(rec[1])), k))


# ======================================================================================
# CROSS-VALIDATION
# ======================================================================================

def perform_cross_validation(
    data,
    model_class,
    algorithm_name='SVD',
    cv_type='shuffle',       # 'shuffle' for thesis 4.1; 'kfold' if needed
    n_splits=10,
    test_size=0.25,
    random_state=42
):
    """
    Performs cross-validation and returns metrics in a DataFrame.

    cv_type:
      - 'shuffle' -> Surprise ShuffleSplit (recommended for thesis 4.1)
      - 'kfold'   -> Surprise KFold
    """

    if cv_type == 'shuffle':
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    elif cv_type == 'kfold':
        cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    else:
        raise ValueError("cv_type must be 'shuffle' or 'kfold'")

    metrics = {
        'RMSE': [],
        'MAE': [],
        'MSE': [],
        'FCP': [],
        'Fit time': [],
        'Test time': []
    }

    for i, (trainset, testset) in enumerate(cv.split(data), 1):
        model = model_class()

        start_fit = time.time()
        model.fit(trainset)
        end_fit = time.time()

        start_test = time.time()
        predictions = model.test(testset)
        end_test = time.time()

        metrics['Fit time'].append(round(end_fit - start_fit, 2))
        metrics['Test time'].append(round(end_test - start_test, 2))

        metrics['RMSE'].append(accuracy.rmse(predictions, verbose=False))
        metrics['MAE'].append(accuracy.mae(predictions, verbose=False))
        metrics['MSE'].append(accuracy.mse(predictions, verbose=False))
        try:
            metrics['FCP'].append(accuracy.fcp(predictions, verbose=False))
        except ValueError:
            # happens when some users have <2 predictions in testset -> no pairs for FCP
            metrics['FCP'].append(np.nan)

    df_metrics = pd.DataFrame({
        'Algorithm': algorithm_name,
        'Metric': list(metrics.keys()),
        'Mean': [np.nanmean(v) for v in metrics.values()],
        'Std': [np.nanstd(v) for v in metrics.values()]
    })

    return df_metrics


# ======================================================================================
# SCORES
# ======================================================================================


def get_actID_score_df(uIDs, actID_lst, actID_qID_dc, uID_scores, all_answers_df):
    """
    Computes the relevance scores for each user-action pair of a single aspect:
    S[u, a] = a_F2[u] * answer[u, q(a)], with a_F2 the user's weight of the aspect
    and q(a) the question of action a.

    Python and numpy numbers (numbers.Real) are accepted. A missing answer (NaN) gives
    S = NaN, and every combination with that action then fails the threshold r_T. A value
    that is not a number (e.g. the string ' ') gives S = 0.

    Parameters:
    - uIDs (list): User IDs
    - actID_lst (list): Action IDs
    - actID_qID_dc (dict): Mapping from actID to corresponding question ID
    - uID_scores (dict): User weight of the aspect, {uID: a_F2}
    - all_answers_df (DataFrame): User questionnaire answers

    Returns:
    - Sc_df (DataFrame): Relevance scores matrix (users x actions)
    """
    Sc_df = pd.DataFrame(index=uIDs, columns=actID_lst)
    for uID in uIDs:
        c_score = uID_scores[uID]
        for actID in actID_lst:
            c_anws = all_answers_df.at[uID, actID_qID_dc[actID]]
            if isinstance(c_score, numbers.Real) and isinstance(c_anws, numbers.Real):
                Sc_df.at[uID, actID] = c_score * c_anws
            else:
                Sc_df.at[uID, actID] = 0
    return Sc_df


def get_score_estimation(uID, act_seq, uID_actID_answers_df, actID_score_df, compat_df, meth_code):
    """
    Estimates a rating score for a sequence of actions for a user.

    Parameters:
    - uID (int/str): User ID
    - act_seq (tuple): Sequence of action IDs
    - uID_actID_answers_df (DataFrame): User answers matrix (users x actions)
    - actID_score_df (DataFrame): User-action score matrix
    - compat_df (DataFrame): Pairwise action compatibility matrix
    - meth_code (str): Method code ('score')

    Returns:
    - float: Estimated score for the action sequence
    """

    if meth_code == 'score':

        # Get single actions score sum
        c_score = 0
        for actID in act_seq:
            c_score += actID_score_df.at[uID, actID]*uID_actID_answers_df.at[uID, actID]

        # Get compatibility score
        if len(act_seq) >= 2:
            act_pairs = list(itertools.combinations(act_seq, 2))
            curr_compat = 1
            for act_pair in act_pairs:
                act1, act2 = act_pair[0], act_pair[1]
                c_comp_score = compat_df.at[act1, act2]
                curr_compat *= c_comp_score
        else:
            curr_compat = 1 # For a single action

        c_r = c_score - (1 - curr_compat)

    else:
        raise ValueError(f"Unknown meth_code: {meth_code!r}")

    return c_r

def get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst=None):
    """
    Prepares a scaled DataFrame of user responses for selected aspects.

    Parameters:
    - all_answers_df (DataFrame): Raw questionnaire responses
    - group_qLst (dict): Aspect to question list mapping
    - aspect_groups_lst (list): Selected aspects to process

    Returns:
    - scl_X_df (DataFrame): Scaled user responses by question; values that are missing
      or not numeric count as 0 before scaling
    """
    all_qs_lst = [qID for group in aspect_groups_lst or [] for qID in group_qLst[group]]
    X_df = all_answers_df[all_qs_lst].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale every question to the [0, 1] range
    scl_X_np = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_df.to_numpy())
    scl_X_df = pd.DataFrame(data=scl_X_np, index=all_answers_df.index, columns=all_qs_lst)
    scl_X_df.index.names = ['uID']
    return scl_X_df


# ======================================================================================
# EXPORT UTILITIES
# ======================================================================================

def save_df_as_latex_table(
    df: pd.DataFrame,
    out_dir: Path,
    filename_stem: str,
    caption: str,
    label: str,
    float_format: str = "{:.4f}",
    index: bool = False
) -> Path:
    """
    Save a DataFrame as a LaTeX table (.tex, booktabs, decimal comma) in out_dir.

    filename_stem: without extension
    caption/label: used inside LaTeX table env
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_stem}.tex"

    latex_tabular = df.to_latex(
        index=index,
        escape=True,  # keep safe for LaTeX
        column_format="l" + "r" * (df.shape[1] - 1 + index),
        float_format=lambda x: float_format.format(x).replace(".", ","),
        longtable=False
    )

    latex_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"{latex_tabular}\n"
        "\\end{table}\n"
    )

    out_path.write_text(latex_table, encoding="utf-8")
    return out_path


def plot_results_overview(results_df, out_stem):
    """Three panels (strict F1, tie-aware F1, coverage) for all rows of the results table."""
    rows = list(results_df["Category"])
    panels = [("F1-mera (strogo)", "F1"), ("F1-mera z izenačenji", "F1_ties"), ("Pokritost uporabnikov", "Coverage")]
    y = np.arange(len(rows))[::-1]
    # Comparison rows (popularity, random) are grey
    colors = ["#9aa3b0" if r.startswith(("Popularnost", "Naključno")) else "#2A6FDB" for r in rows]
    # Separators between blocks: a block starts with its SVD row (M3-M5 share one block)
    starts = [i for i, r in enumerate(rows) if i and r.startswith(("S kontekstom", "M3"))]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.46 * len(rows) + 0.7), sharey=True)
    for ax, (title, col) in zip(axes, panels):
        vals = pd.to_numeric(results_df[col], errors="coerce")
        for yi, v, c in zip(y, vals, colors):
            if pd.isna(v):
                ax.text(0.02, yi, "ni definirano", va="center", fontsize=8.5, color="#5b6474", style="italic")
                continue
            ax.barh(yi, v, height=0.62, color=c)
            ax.text(v + 0.02, yi, f"{v:.3f}".replace(".", ","), va="center", fontsize=8.8)
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold")
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0,25", "0,5", "0,75", "1"], fontsize=8.5, color="#5b6474")
        ax.grid(axis="x", color="#e6e8ec", lw=0.7)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        for i in starts:
            ax.axhline(y[i] + 0.5, color="#cfd4db", lw=0.8, ls=(0, (3, 3)))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows, fontsize=9.5)
    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================================
# EVALUATION UTILITIES
# ======================================================================================

def build_reference(D_triplets, top_n, gt_filter=None):
    """
    Per-user reference from the test triplets (uid, item, rating).

    Returns {uid: (relevant, relevant_ties)}: the top_n rated items of the user (ties
    broken by the item's activity IDs), and all the user's items rated at least as high
    as the last of them. gt_filter is an optional predicate on item; users without a
    matching item have no reference and are left out.
    """
    user_gt = defaultdict(list)
    for uid, item, rating in D_triplets:
        if gt_filter is None or gt_filter(item):
            user_gt[uid].append((item, rating))

    reference = {}
    for uid, rated in user_gt.items():
        top_gt = sorted(rated, key=lambda x: (-x[1], tuple(x[0])))[:top_n]
        r_cut = top_gt[-1][1]
        reference[uid] = ({a for a, _ in top_gt}, {a for a, r in rated if r >= r_cut})
    return reference


def evaluate_topk(D_triplets, recs, k, top_n, gt_filter=None):
    """
    Top-k ranking metrics against the per-user reference (top_n rated test items).

    Parameters:
    - D_triplets: (uid, item, rating) test triplets; item is a complete activity tuple
    - recs: (uid, item, score) recommendations; items are compared as whole tuples
    - k (int): Number of recommendations to evaluate per user
    - top_n (int): Reference size per user (smaller if the user has fewer items)
    - gt_filter (callable): Optional predicate on item; only matching items
      enter the reference

    Every user with a nonempty reference is evaluated, including users without
    recommendations (P = R = F1 = 0). Users without a reference have undefined
    recall and are outside the evaluation population.

    F1_ties (tie-aware): every item rated at least as high as the last reference
    item is relevant; the recall denominator stays the reference size, and the hits are
    capped at it (so F1_ties <= 1 also when k > top_n).

    Returns:
    - dict(P, R, F1, F1_ties, n_users): unrounded means over users
    """
    user_recs = defaultdict(list)
    for uid, item, score in recs:
        user_recs[uid].append((item, float(score)))

    def prf(tp, n_predicted, n_reference):
        p = tp / n_predicted if n_predicted else 0.0
        r = tp / n_reference
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    rows = []
    for uid, (relevant, relevant_ties) in build_reference(D_triplets, top_n, gt_filter).items():
        ranked = sorted(user_recs.get(uid, []), key=lambda x: (-x[1], tuple(x[0])))
        predicted = {a for a, _ in ranked[:k]}

        strict = prf(len(predicted & relevant), len(predicted), len(relevant))
        ties = prf(min(len(predicted & relevant_ties), len(relevant)), len(predicted), len(relevant))
        rows.append((*strict, ties[2]))

    return _mean_metrics(rows)


def expected_random_topk(D_triplets, candidates, k, top_n):
    """
    Expected metrics of a uniformly random ranking of each user's candidates (analytic, no seed).

    For a user with n candidates, the random list B has b = min(k, n) items. With a reference
    A of size a, of which h items are among the candidates, E|A & B| = b * h / n. Precision,
    recall and F1 = 2|A & B| / (a + b) are linear in |A & B|, so
        E[P] = h / n,   E[R] = b * h / (n * a),   E[F1] = 2 * b * h / (n * (a + b)).
    The tie-aware F1 replaces h by the number of candidates in the tie-extended reference.
    Its hits are capped at a (as in evaluate_topk); if b > a, the expectation of the capped
    hypergeometric count is computed exactly. If the candidates are the reference pool itself,
    then a = b = h = m = min(k, n) (for k = top_n) and E[F1] = m / n.

    Parameters:
    - D_triplets: (uid, item, rating) test triplets that define the reference
    - candidates: (uid, item, rating) test triplets that may be recommended

    Returns:
    - dict(P, R, F1, F1_ties, n_users, n_covered, avg_rating); n_covered is the number of
      users with at least one candidate, avg_rating the expected mean rating of the returned
      items (NaN if no user has a candidate)
    """
    user_cand = defaultdict(list)
    for uid, item, rating in candidates:
        user_cand[uid].append((item, rating))

    rows = []
    for uid, (relevant, relevant_ties) in build_reference(D_triplets, top_n).items():
        items = {item for item, _ in user_cand.get(uid, [])}
        n, a = len(items), len(relevant)
        if n == 0:
            rows.append((0.0, 0.0, 0.0, 0.0))
            continue
        b = min(k, n)
        h, h_ties = len(relevant & items), len(relevant_ties & items)
        if b > a:
            # E[min(X, a)] for X ~ hypergeometric(n items, h_ties relevant, b drawn)
            hits_ties = sum(min(x, a) * comb(h_ties, x) * comb(n - h_ties, b - x) for x in range(b + 1)) / comb(n, b)
            f1_ties = 2 * hits_ties / (a + b)
        else:
            f1_ties = 2 * b * h_ties / (n * (a + b))
        rows.append((h / n, b * h / (n * a), 2 * b * h / (n * (a + b)), f1_ties))

    n_returned = [min(k, len(rated)) for rated in user_cand.values()]
    mean_ratings = [np.mean([r for _, r in rated]) for rated in user_cand.values()]
    avg_rating = float(np.average(mean_ratings, weights=n_returned)) if user_cand else float("nan")
    return {**_mean_metrics(rows), "n_covered": len(user_cand), "avg_rating": avg_rating}


def _mean_metrics(rows):
    """Means over users of the per-user (P, R, F1, F1_ties) rows."""
    if not rows:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "F1_ties": 0.0, "n_users": 0}
    p, r, f1, f1_ties = (float(np.mean(column)) for column in zip(*rows))
    return {"P": p, "R": r, "F1": f1, "F1_ties": f1_ties, "n_users": len(rows)}


def evaluate_groups(D_triplets, recs_by_group, actID_to_group, k, top_n, n_test_users, layer_name):
    """
    Evaluates each context group against its own reference (same all-member
    membership rule for candidates and reference). Values are not rounded.

    Returns:
    - df_per_group: per-group metrics
    - avg_dc: average across groups, weighted by NumUsers; AverageScore is the unweighted
      mean of the groups, AverageReferenceRating the mean over all returned recommendations
    """
    ratings = {(uid, item): rating for uid, item, rating in D_triplets}
    rows = []
    for g, recs in recs_by_group.items():
        m = evaluate_topk(D_triplets, recs, k, top_n,
                          gt_filter=lambda item: item_in_group(item, g, actID_to_group))
        rows.append({
            "Layer": layer_name,
            "Group": g,
            "Precision": m["P"],
            "Recall": m["R"],
            "F1": m["F1"],
            "F1_ties": m["F1_ties"],
            "Coverage": coverage(recs, n_test_users),
            "AverageScore": average_score(recs),
            "AverageReferenceRating": average_reference_rating(recs, ratings),
            "NumRecs": len(recs),
            "NumUsers": m["n_users"],
        })

    df = pd.DataFrame(rows)
    avg_dc = {c: float(np.average(df[c], weights=df["NumUsers"]))
              for c in ("Precision", "Recall", "F1", "F1_ties")}
    avg_dc["AverageScore"] = float(df["AverageScore"].mean())
    avg_dc["AverageReferenceRating"] = average_reference_rating(
        [rec for recs in recs_by_group.values() for rec in recs], ratings)
    return df, avg_dc


def context_breakdown(D_triplets_ctx, recs_by_method, users, users_feasible_anywhere, k, top_n):
    """
    Splits the test users by the number of feasible test items (context reference):
    none, 1 to k, more than k, and evaluates each subset.

    Parameters:
    - D_triplets_ctx: test triplets that are feasible in the user's context
    - recs_by_method: {name: recommendations}
    - users: all test users
    - users_feasible_anywhere: users with a feasible item anywhere in D_lst

    Returns:
    - DataFrame with one row per subset (metrics are NaN without a reference)
    """
    n_feasible = Counter(uid for uid, _, _ in D_triplets_ctx)
    none = {u for u in users if n_feasible[u] == 0}
    subsets = [
        ("0, skladna kombinacija le v razvojnem delu", none & users_feasible_anywhere),
        ("0, brez skladne kombinacije v celotnem naboru", none - users_feasible_anywhere),
        (f"1–{k}", {u for u in users if 1 <= n_feasible[u] <= k}),
        (f"več kot {k}", {u for u in users if n_feasible[u] > k}),
        ("skupaj vsaj 1", {u for u in users if n_feasible[u] > 0}),
    ]
    rows = []
    for name, subset in subsets:
        reference = [t for t in D_triplets_ctx if t[0] in subset]
        row = {"FeasibleTestItems": name, "NumUsers": len(subset)}
        metrics = {method: evaluate_topk(reference, recs, k, top_n) for method, recs in recs_by_method.items()}
        metrics["Random"] = expected_random_topk(reference, reference, k, top_n) if reference else _mean_metrics([])
        for metric in ("F1", "F1_ties"):
            for method, m in metrics.items():
                row[f"{metric}_{method}"] = m[metric] if reference else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(recs, n_test_users):
    """Share of test users with at least one recommendation."""
    return len({uid for uid, _, _ in recs}) / n_test_users


def average_score(recs):
    """Mean score of the returned recommendations (0 if there are none)."""
    return float(np.mean([s for _, _, s in recs])) if recs else 0.0


def average_reference_rating(recs, ratings):
    """Mean reference (test) rating of the returned recommendations (0 if there are none)."""
    return float(np.mean([ratings[uid, item] for uid, item, _ in recs])) if recs else 0.0


def round_floats(df, digits=3):
    """Rounds the float cells of a table for export; other cells are kept."""
    return df.map(lambda v: round(float(v), digits) if isinstance(v, float) else v)
