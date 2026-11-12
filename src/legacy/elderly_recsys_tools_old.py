"""
elderly_recsys_tools.py
=======================

Tools for the "Situation-aware elderly daily activity recommender system" project.

Key features:
- Build D_lst (sparse user-item matrix with optional context)
- Context feasibility checks
- Surprise-based recommendation helper
- Cross-validation (ShuffleSplit is default for thesis)
- Precision/Recall/F1 evaluation (with optional context-aware ground truth)
- Export tables to LaTeX

"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Iterable, Optional

# File Tools for elderly recommender system
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from surprise import SVD, accuracy
from surprise.model_selection import KFold, ShuffleSplit
import random
import time 
from tqdm import tqdm
from pathlib import Path


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
                        # pridobi kontekst za vsako aktivnost v sekvenci
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
                    entry[3] = (min_target + max_target) / 2  # če so vse ocene enake

    return D_lst

# ======================================================================================
# CONTEXT
# ======================================================================================

def get_random_context(all_contexts):
    """
    Selects a random simplified context from a list of full activity contexts.

    Parameters:
    - all_contexts (list of dict): A list of context dictionaries, where each dictionary contains
                                   the full context description of an activity.

    Returns:
    - c_cntx (dict): A context dictionary with a randomly selected time (C_T) and place (C_P)
                     from a randomly selected full context. Useful for testing context matching.
    """

    full_cntx = random.choice(all_contexts)
    C_T = random.choice(['act_C_T1', 'act_C_T2', 'act_C_T3'])
    C_P = random.choice(['act_C_P1', 'act_C_P2', 'act_C_P3'])

    c_cntx = {'qID': full_cntx['qID'], 'C_T': full_cntx[C_T], 'C_P': full_cntx[C_P], 'C_A':''}
    
    return c_cntx

def get_one_random_context(full_cntx):
    """
    Selects a random context (time and place) from a full context definition for a single activity.

    Parameters:
    - full_cntx (dict): Dictionary containing full context fields for one activity,
                        including keys like 'qID', 'act_C_T1', 'act_C_T2', 'act_C_T3', 'act_C_P1', 'act_C_P2', 'act_C_P3'.

    Returns:
    - c_cntx (dict): A simplified context dictionary with one randomly selected time (C_T)
                     and one place (C_P), and empty action field (C_A), e.g.:
                     {
                         'qID': 'AB4_1',
                         'C_T': 'dopoldne',
                         'C_P': 'doma',
                         'C_A': ''
                     }
    """
    if full_cntx is None:
        full_cntx = {}

    C_T = random.choice(['act_C_T1', 'act_C_T2', 'act_C_T3'])
    C_P = random.choice(['act_C_P1', 'act_C_P2', 'act_C_P3'])

    c_cntx = {'qID': full_cntx['qID'], 'act_C_T': full_cntx[C_T], 'act_C_P': full_cntx[C_P], 'act_C_A':''}
    
    return c_cntx


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

    time_ok = user_C_T in C_Ts if user_C_T else False
    place_ok = user_C_P in C_Ps if user_C_P else False

    # relax "kjerkoli"
    if relax_kjerkoli:
        if user_C_P == 'kjerkoli':
            place_ok = True
        if 'kjerkoli' in C_Ps:
            place_ok = True

    return bool(time_ok and place_ok)


def is_action_in_context_group(actID: str, group_id: int, actID_to_group: dict) -> bool:
    if actID not in actID_to_group:
        return False
    return int(actID_to_group[actID]) == int(group_id)


# ======================================================================================
# RECOMMENDATIONS
# ======================================================================================

def get_recommendations(uID, 
                        n_recommendations=20, 
                        D_lst=None, 
                        trainset=None, 
                        model=None, 
                        context=None, 
                        actID_context_dc=None):
    """
    Returns top-N recommended actions for a given user.
    
    You can use:
    - D_lst for precomputed recommendations (triplets: user_id, item_id, score)
    - model + trainset for matrix factorization-based recommendations
    - Optionally apply context filtering (if context & actID_context_dc provided)

    Parameters:
    - uID (int or str): ID of the user
    - n_recommendations (int): Number of top results to return
    - D_lst (list of tuples): Optional. Precomputed (uID, item, score) list
    - trainset (Surprise Trainset): Optional. Surprise trainset object
    - model (Surprise model): Optional. Trained Surprise model
    - context (dict): Optional. Context to filter actions (e.g., {'C_T': 'dopoldne'})
    - actID_context_dc (dict): Optional. Dictionary with context info for each action

    Returns:
    - List of top-N recommendations: [(uID, item, score), ...]
    - First returns top-20 candidates, then you can filter afterwards
    """

    # Method 1: Use precomputed D_lst
    if D_lst is not None:
        user_entries = [x for x in D_lst if x[0] == uID]
        sorted_entries = sorted(user_entries, key=lambda x: x[3], reverse=True)
        return sorted_entries[:n_recommendations]

    # Method 2: Use trained model for prediction
    elif model is not None and trainset is not None:
        all_iids = trainset._raw2inner_id_items.keys()
        predictions = []

        for raw_iid in all_iids:
            est = model.predict(uid=uID, iid=raw_iid).est
            predictions.append((uID, raw_iid, est))

        sorted_predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
        return sorted_predictions[:n_recommendations]

    else:
        raise ValueError("Provide either D_lst or both model and trainset.")

def evaluate_recommender_metrics(D_lst, best_act_trp_lst, top_n_groundtruth=20, k_eval=5):
    """
    Evaluate recommender metrics (Precision, Recall, F1) for action sequences.
    
    Parameters:
    - D_lst: List of [uid, act_seq, context_seq, score] (4 elements per entry)
    - best_act_trp_lst: List of recommended [uid, act_seq, score]
    - top_n_groundtruth: Number of top items to consider as ground truth
    - k_eval: Number of recommendations to evaluate (top-K)
    
    Returns:
    - avg_p, avg_r, avg_f: Average precision, recall, F1 scores
    """
    user_action_scores = defaultdict(list)
    for entry in D_lst:
        uid, act_seq, context_seq, score = entry[0], entry[1], entry[2], entry[3]
        user_action_scores[uid].append((tuple(act_seq), score))

    ground_truth_dict = {}
    for uid, scored_seqs in user_action_scores.items():
        top_gt = sorted(scored_seqs, key=lambda x: x[1], reverse=True)[:top_n_groundtruth]
        ground_truth_dict[uid] = set([tuple(a) for a, _ in top_gt])

    recommended_dict = defaultdict(list)
    for uid, act_seq, score in best_act_trp_lst:
        recommended_dict[uid].append((tuple(act_seq), score))

    precision_list, recall_list, f1_list = [], [], []

    for uid, recs in recommended_dict.items():
        top_k = sorted(recs, key=lambda x: x[1], reverse=True)[:k_eval]
        predicted = set([act for act, _ in top_k])
        actual = ground_truth_dict.get(uid, set())
        if not actual:
            continue

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    if precision_list:
        avg_p = round(sum(precision_list) / len(precision_list), 3)
        avg_r = round(sum(recall_list) / len(recall_list), 3)
        avg_f = round(sum(f1_list) / len(f1_list), 3)
    else:
        avg_p = avg_r = avg_f = 0.0

    return avg_p, avg_r, avg_f


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
# SCORES / COMPATIBILITY
# ======================================================================================

def get_context(act_C_T1, act_C_T2, act_C_T3):
    """
    Constructs a context string from three contextual time values.

    Parameters:
    - act_C_T1, act_C_T2, act_C_T3 (str or NaN): Contextual descriptors

    Returns:
    - str: Concatenated context string or 'nd' if none are valid
    """

    valid_contexts = {'dopoldne', 'popoldne', 'zvečer'}  
    context = []

    for c in [act_C_T1, act_C_T2, act_C_T3]:
        if pd.isna(c):
            continue
        c = c.lower().strip()
        if c in valid_contexts:
            context.append(c)
    if not context:  
        return 'nd'
    return '-'.join(sorted(context))  

# @brief copute single action score = rating compatibility
def get_actID_score_df(uIDs, actID_lst, actID_qID_dc, uID_scores_dc, all_answers_df, aspect_groups_lst, meth_code):
    """
    Computes the relevance scores for each user-action pair.

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
        # Normalisation: to do
    return Sc_df

# @brief: compute dataframe of action pair compatibilities
def get_actIDPair_compat_df(actID_lst, qID_Group_dc, actID_qID_dc):
    """
    Calculates compatibility between pairs of actions based on question group similarity.

    Parameters:
    - actID_lst (list): Action IDs
    - qID_Group_dc (dict): Question ID to group mapping
    - actID_qID_dc (dict): Action ID to question ID mapping

    Returns:
    - Cmp_df (DataFrame): Compatibility score matrix
    """

    Cmp_df = pd.DataFrame(index=actID_lst, columns=actID_lst)
    n = len(actID_lst)

    for act1_i in range(n):
        for act2_i in range(n):
            act1, act2 = actID_lst[act1_i], actID_lst[act2_i]
            qID1, qID2 = actID_qID_dc[act1], actID_qID_dc[act2]
            group1, group2 = qID_Group_dc[qID1], qID_Group_dc[qID2]
            Cmp_df.at[act1, act2] = 0
            if group1 == group2:
                Cmp_df.at[act1, act2] = 0.1
            elif group1 != group2:
                Cmp_df.at[act1, act2] = 0.9

    # Normalisation: to do

    return Cmp_df


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

        # Get ompatibitliy score
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

    return c_r

# @brief get anwssers by users
def get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst=[]):
    """
    Prepares a scaled DataFrame of user responses for selected aspects.

    Parameters:
    - all_answers_df (DataFrame): Raw questionnaire responses
    - group_qLst (dict): Aspect to question list mapping
    - aspect_groups_lst (list): Selected aspects to process

    Returns:
    - scl_X_df (DataFrame): Scaled user responses by question
    """
    all_qs_lst = []
    for group in aspect_groups_lst:
        all_qs_lst = all_qs_lst + group_qLst[group]

    X_df = pd.DataFrame(index=all_answers_df.index, columns=all_qs_lst)
    X_df.fillna(0, inplace=True)
    all_answers_nums_df = all_answers_df.replace(r' ', np.nan, regex=True).astype(float)

    
    if aspect_groups_lst != []:
        for group in aspect_groups_lst:
            c_X_df = pd.DataFrame(index=all_answers_df.index)
            for qID in group_qLst[group]:
                c_X_df[qID] = all_answers_nums_df[qID]

            scaler = MinMaxScaler(feature_range=(0, 1))
            c_scl_X_np = scaler.fit_transform(c_X_df.to_numpy())
            c_scl_X_np = pd.DataFrame(data=c_scl_X_np, index=all_answers_df.index, columns=group_qLst[group])
            
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


# ======================================================================================
# EVALUATION UTILITIES
# ======================================================================================

def normalize_act_id(x):
    return x[0] if isinstance(x, tuple) else x

def build_D_triplets_from_Dlst(D_lst):
    """Keep only single-action entries: (uid, (actID,), rating)"""
    D_triplets = []
    for uid, act_seq, context_seq, rating in D_lst:
        if isinstance(act_seq, (list, tuple)) and len(act_seq) == 1:
            D_triplets.append((uid, tuple(act_seq), float(rating)))
    return D_triplets

def evaluate_recommender_metrics_filtered_groundtruth(
    D_triplets,
    rec_triplets,
    *,
    top_n_groundtruth=20,
    k_eval=5,
    groundtruth_filter_fn=None
):
    """
    Ranking metrics with OPTIONAL filtered ground truth (important for M3/M4/M5).
    D_triplets: (uid, (actID,), rating)
    rec_triplets: (uid, (actID,), score)
    """
    user_gt = defaultdict(list)
    for uid, act_seq, rating in D_triplets:
        if groundtruth_filter_fn is None or groundtruth_filter_fn(act_seq):
            user_gt[uid].append((act_seq, rating))

    gt_dict = {}
    for uid, seqs in user_gt.items():
        top_gt = sorted(seqs, key=lambda x: x[1], reverse=True)[:top_n_groundtruth]
        gt_dict[uid] = set([a for a, _ in top_gt])

    rec_dict = defaultdict(list)
    for uid, act_seq, score in rec_triplets:
        rec_dict[uid].append((act_seq, float(score)))

    precision_list, recall_list, f1_list = [], [], []

    for uid, recs in rec_dict.items():
        if uid not in gt_dict or len(gt_dict[uid]) == 0:
            continue

        top_k = sorted(recs, key=lambda x: x[1], reverse=True)[:k_eval]
        predicted = set([a for a, _ in top_k])
        actual = gt_dict[uid]

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    if len(precision_list) == 0:
        return 0.0, 0.0, 0.0

    return (
        round(float(np.mean(precision_list)), 3),
        round(float(np.mean(recall_list)), 3),
        round(float(np.mean(f1_list)), 3),
    )



# Fallback if you don't have erst.is_action_in_context_group
def is_action_in_context_group_local(actID: str, group_id: int, actID_to_group: dict) -> bool:
    if actID not in actID_to_group:
        return False
    try:
        return int(actID_to_group[actID]) == int(group_id)
    except Exception:
        return str(actID_to_group[actID]) == str(group_id)



# =============================================================================
# Random context logic for 4.2 (switchable)
# =============================================================================

def build_context_pool(actID_context_dc: dict):
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
    return {
        "C_T": random.choice(C_T_pool) if C_T_pool else "nd",
        "C_P": random.choice(C_P_pool) if C_P_pool else "nd",
    }
