"""
elderly_recsys_tools.py
=======================

Tools for the "Situation-aware elderly daily activity recommender system" project.

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
import random
import time
from collections import defaultdict
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
    - normalize (bool): If True, normalize all ratings to range [0, 5]

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

def is_action_context_feasibleQ(
    actID: str,
    cntx: dict,
    actID_context_dc: dict,
    *,
    relax_kjerkoli: bool = True,
    relax_nd: bool = True
) -> bool:
    """
    Checks if actID is feasible in a given context.

    cntx expected keys:
      - 'C_T' (time label)
      - 'C_P' (place label)

    actID_context_dc[actID] contains:
      - act_C_T1..3, act_C_P1..3
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

    # relax 'nd' (unknown) => treat as feasible
    if relax_nd and (user_C_T == 'nd' or user_C_P == 'nd'):
        return True

    # "kadarkoli" means no time restriction, in the activity definition or the selected context.
    time_ok = bool(user_C_T) and (user_C_T == 'kadarkoli' or user_C_T in C_Ts or 'kadarkoli' in C_Ts)
    place_ok = user_C_P in C_Ps if user_C_P else False

    # relax "kjerkoli"
    if relax_kjerkoli:
        if user_C_P == 'kjerkoli':
            place_ok = True
        if 'kjerkoli' in C_Ps:
            place_ok = True

    return bool(time_ok and place_ok)


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
    """Draws one simulated context (time, place); uses the global `random` state."""
    return {
        "C_T": random.choice(C_T_pool) if C_T_pool else "nd",
        "C_P": random.choice(C_P_pool) if C_P_pool else "nd",
    }


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


def get_actID_score_df(uIDs, actID_lst, actID_qID_dc, uID_scores_dc, all_answers_df, aspect_groups_lst, meth_code):
    """
    Computes the relevance scores for each user-action pair:
    S[u, a] = a_F2[u] * answer[u, q(a)], with a_F2 the user's activity score
    and q(a) the question of action a (0 if either value is not numeric).

    Parameters:
    - uIDs (list): User IDs
    - actID_lst (list): Action IDs
    - actID_qID_dc (dict): Mapping from actID to corresponding question ID
    - uID_scores_dc (dict): User scores by aspect
    - all_answers_df (DataFrame): User questionnaire answers
    - aspect_groups_lst (list): Aspects to use
    - meth_code (str): Scoring method (e.g., 'score')

    Returns:
    - Sc_df (DataFrame): Relevance scores matrix (users x actions)
    """

    Sc_df = pd.DataFrame(index=uIDs, columns=actID_lst)

    for group in aspect_groups_lst:
        c_uID_activity_scores_dc = uID_scores_dc[group]
    
        for uID in uIDs:
            for actID in actID_lst:
                qID = actID_qID_dc[actID]
                c_score = c_uID_activity_scores_dc[uID]
                c_anws = all_answers_df.at[uID, qID]
                if isinstance(c_score, (int, float)) and isinstance(c_anws, (int, float)):
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
    - scl_X_df (DataFrame): Scaled user responses by question
    """
    aspect_groups_lst = aspect_groups_lst or []

    all_qs_lst = []
    for group in aspect_groups_lst:
        all_qs_lst = all_qs_lst + group_qLst[group]

    X_df = pd.DataFrame(index=all_answers_df.index, columns=all_qs_lst)
    X_df.fillna(0, inplace=True)
    all_answers_nums_df = all_answers_df.replace(r' ', np.nan, regex=True).astype(float)

    for group in aspect_groups_lst:
        c_X_df = pd.DataFrame(index=all_answers_df.index)
        for qID in group_qLst[group]:
            c_X_df[qID] = all_answers_nums_df[qID]
        X_df = X_df.add(c_X_df, fill_value=0)

    # Scale all answers to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scl_X_np = scaler.fit_transform(X_df.to_numpy())
    scl_X_df = pd.DataFrame(data=scl_X_np, index = all_answers_df.index, columns=all_qs_lst)
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
    Save a DataFrame as a LaTeX table (.tex) in out_dir.

    filename_stem: without extension
    caption/label: used inside LaTeX table env
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{filename_stem}.tex"

    latex_tabular = df.to_latex(
        index=index,
        escape=True,  # keep safe for LaTeX
        float_format=lambda x: float_format.format(x) if isinstance(x, (float, int)) else str(x),
        longtable=False
    )

    latex_table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex_tabular}\n"
        "\\end{table}\n"
    )

    out_path.write_text(latex_table, encoding="utf-8")
    return out_path


def plot_results_overview(table_45, out_stem):
    """Three panels (strict F1, tie-aware F1, coverage) for all rows of table 4.5."""
    rows = list(table_45["Category"])
    panels = [("F1-mera (strogo)", "F1"), ("F1-mera z izenačenji", "F1_ties"), ("Pokritost uporabnikov", "Coverage")]
    y = np.arange(len(rows))[::-1]
    colors = ["#9aa3b0" if "Popularnost" in r else "#2A6FDB" for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    for ax, (title, col) in zip(axes, panels):
        vals = pd.to_numeric(table_45[col], errors="coerce")
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
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        # dashed separators between A / B / C blocks
        for yb in (y[1] - 0.5, y[3] - 0.5):
            ax.axhline(yb, color="#cfd4db", lw=0.8, ls=(0, (3, 3)))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows, fontsize=9.5)
    fig.tight_layout(w_pad=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================================
# EVALUATION UTILITIES
# ======================================================================================

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
    item is relevant; the recall denominator stays the reference size.

    Returns:
    - dict(P, R, F1, F1_ties, n_users): unrounded means over users
    """
    user_gt = defaultdict(list)
    for uid, item, rating in D_triplets:
        if gt_filter is None or gt_filter(item):
            user_gt[uid].append((item, rating))

    user_recs = defaultdict(list)
    for uid, item, score in recs:
        user_recs[uid].append((item, float(score)))

    def prf(tp, n_predicted, n_reference):
        p = tp / n_predicted if n_predicted else 0.0
        r = tp / n_reference
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    def ranking_key(x):
        return (-x[1], tuple(x[0]))

    rows = []
    for uid, rated in user_gt.items():
        top_gt = sorted(rated, key=ranking_key)[:top_n]
        r_cut = top_gt[-1][1]
        relevant = {a for a, _ in top_gt}
        relevant_ties = {a for a, r in rated if r >= r_cut}
        predicted = {a for a, _ in sorted(user_recs.get(uid, []), key=ranking_key)[:k]}

        strict = prf(len(predicted & relevant), len(predicted), len(relevant))
        ties = prf(len(predicted & relevant_ties), len(predicted), len(relevant))
        rows.append((*strict, ties[2]))

    if not rows:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "F1_ties": 0.0, "n_users": 0}

    p, r, f1, f1_ties = (float(np.mean(column)) for column in zip(*rows))
    return {"P": p, "R": r, "F1": f1, "F1_ties": f1_ties, "n_users": len(rows)}


def evaluate_groups(D_triplets, recs_by_group, actID_to_group, k, top_n, n_test_users, layer_name):
    """
    Evaluates each context group against its own reference (same all-member
    membership rule for candidates and reference).

    Returns:
    - df_per_group: per-group metrics
    - avg_dc: average across groups, weighted by NumUsers (AverageScore unweighted)
    """
    rows = []
    for g, recs in recs_by_group.items():
        m = evaluate_topk(D_triplets, recs, k, top_n,
                          gt_filter=lambda item: item_in_group(item, g, actID_to_group))
        rows.append({
            "Layer": layer_name,
            "Group": g,
            "Precision": round(m["P"], 3),
            "Recall": round(m["R"], 3),
            "F1": round(m["F1"], 3),
            "F1_ties": round(m["F1_ties"], 3),
            "Coverage": round(coverage(recs, n_test_users), 3),
            "AverageScore": round(average_score(recs), 3),
            "NumRecs": len(recs),
            "NumUsers": m["n_users"],
        })

    df = pd.DataFrame(rows)
    avg_dc = {c: round(float(np.average(df[c], weights=df["NumUsers"])), 3)
              for c in ("Precision", "Recall", "F1", "F1_ties")}
    avg_dc["AverageScore"] = round(float(df["AverageScore"].mean()), 3)
    return df, avg_dc


def coverage(recs, n_test_users):
    """Share of test users with at least one recommendation."""
    return len({uid for uid, _, _ in recs}) / n_test_users


def average_score(recs):
    """Mean score of the returned recommendations (0 if there are none)."""
    return float(np.mean([s for _, _, s in recs])) if recs else 0.0
