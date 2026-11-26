"""
ELDERLY RECOMMENDER SYSTEM PIPELINE
===================================

This script runs the experiment of the master's thesis "Kontekstualiziran priporočilni
sistem za dnevne aktivnosti starejših": it recommends combinations of daily activities to
elderly users with matrix factorization (SVD), without context, with a simulated user
context, and within context groups (M3/M4/M5). All functions are in elderly_recsys_tools.py.

STEP 0: Import required libraries
STEP 1: Configuration & settings
STEP 2: Load user and activity data from Excel
STEP 3: Build dictionaries from context definitions
STEP 4: Select actions based on chosen aspects
STEP 5: Generate combinations of actions
STEP 6: Load or build the D_lst data matrix
STEP 7: Held-out split and comparison of algorithms (Surprise)
STEP 8: Hyperparameter tuning (GridSearch for SVD)
STEP 9: Cross-validation for tuned SVD
STEP 10: Final model training and matrix factorization
STEP 11: Generate recommendations (4.1/4.2/4.3) and export
STEP 12: Evaluate recommendations and export the results
"""

#%% STEP 0: IMPORT REQUIRED LIBRARIES
##==================================================================================
print("========== STEP 0: IMPORTING LIBRARIES ==========")

import datetime
import pickle
import random as python_random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from surprise import BaselineOnly, Dataset, KNNBasic, NMF, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, KFold
from tqdm import tqdm

import elderly_recsys_tools as erst

#%% STEP 1: CONFIGURATION & SETTINGS
##==================================================================================
print("========== STEP 1: GLOBAL SETTINGS ==========")
np.random.seed(42)
python_random.seed(42)

# ========== ENVIRONMENT & PATHS ==========
BASE_DIR = Path(__file__).resolve().parents[1]

data_path = BASE_DIR / 'data'
context_path = data_path / 'context'
D_lst_path = data_path / 'D_lst'
questionnaires_path = data_path / 'questionnaires'
scores_path = data_path / 'scores'

outputs_path = BASE_DIR / 'outputs' / 'heldout_evaluation'
evaluation_path = outputs_path / 'evaluation'
tabs_path = outputs_path / 'latex' / 'tabs'
figs_path = outputs_path / 'latex' / 'figs'
recommendations_path = outputs_path / 'recommendations'
for folder in (evaluation_path, tabs_path, figs_path, recommendations_path):
    folder.mkdir(parents=True, exist_ok=True)
print('Evaluation: held-out pairs; grouped metrics use group-specific references.')

# Generate timestamp for output versioning (YYYYMMDD_HHMMSS)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ========== EXECUTION MODES ==========
test_mode = False                # True = smoke test on a slice of D_lst; False for final results
USE_EXISTING_D_LST = True       # Load existing D_lst or generate new one
USE_RANDOM_CONTEXT_42 = True    # True = one random context per user, False = FIXED_CONTEXT_42 for all
SKIP_TUNING = True              # True = skip STEP 7-9 comparisons and use BEST_PARAMS (selected in run 20261015_150006)
BEST_PARAMS = {'random_state': 42, 'n_factors': 150, 'n_epochs': 30, 'lr_all': 0.005, 'reg_all': 0.02}
if test_mode:
    print("TESTNI ZAGON: skrajšan nabor; ti rezultati niso za magistrsko besedilo.")

# ========== DATA PROCESSING PARAMETERS ==========
act_max_len = 3                 # Maximum allowed number of actions in a combination
meth_code = 'score'             # Method for relevance computation
r_T = 0.3                       # Threshold for relevance inclusion
TEST_D_LST_MAX_ROWS = 100000    # Max rows to load when test_mode=True

# ========== MODEL & EVALUATION PARAMETERS ==========
n_splits = 10                   # Number of CV splits/folds; not changed by test_mode (test_mode only slices D_lst and shrinks the SVD grid)
test_size = 0.25                # Proportion of dataset for test split
top_k_final_recommendations = 5 # Number of final context-aware recommendations
top_n_groundtruth = 5           # Top ground-truth items for evaluation

# ========== CONTEXT & RECOMMENDATION SETTINGS ==========
FIXED_CONTEXT_42 = {'C_T': 'dopoldne', 'C_P': 'doma'}  # Default context if not randomized

# ========== EXISTING D_LST PATH ==========
EXISTING_D_LST_PATH = D_lst_path / 'D_lst_full_20260202_150312.pkl'


#%% STEP 2: LOAD USER AND ACTIVITY DATA FROM EXCEL
##==================================================================================
print("========== STEP 2: LOADING DATA FROM EXCEL FILES ==========")

# Answers to the activity questionnaire (the only one the experiment uses): -1 becomes NaN,
# columns get the prefix 'Ac_'
all_answers_df = pd.read_excel(questionnaires_path / 'Activities.xlsx', sheet_name='AllAnswers',
                               index_col='S4').replace(-1, np.nan).add_prefix('Ac_')

# Load user-specific scores and weights (how suitable each activity is for each user)
scores_and_wgt_df = pd.read_excel(scores_path / 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')

# Load activity definitions (time and place conditions, context groups) and the
# handcrafted compatibility matrix (values between 0 and 1) from the same workbook
with pd.ExcelFile(context_path / 'ActivityContextGen_v12.xlsx') as context_book:
    action_df = context_book.parse('ActionLst').replace(-1, np.nan)
    action_compat_df = context_book.parse('ActionCompatibilities', index_col=0)

#%% STEP 3: BUILD DICTIONARIES FROM CONTEXT DEFINITIONS
##==================================================================================
print("========== STEP 3: BUILDING DICTIONARIES & STRUCTURES ==========")

# Extract user IDs from the scores DataFrame
uIDs = list(np.sort(scores_and_wgt_df.index))

# Question IDs (questionnaire columns) of the activity aspect
activity_qs = ['Ac_AB4_1', 'Ac_AB4_2', 'Ac_AB4_3', 'Ac_AB4_4', 'Ac_AB4_5', 'Ac_AB4_8']
group_qLst = {'activity': activity_qs}

# The question ID is given only in the first row of each question
action_qIDs = action_df['qID'].ffill()

actID_qID_dc = dict(zip(action_df['actID'], action_qIDs))  # Maps actID to qID

# Maps actID to its context (time and place descriptors)
actID_context_dc = {}
for qID, (_, row) in zip(action_qIDs, action_df.iterrows()):
    actID_context_dc[row['actID']] = {
        'qID': qID,
        'act_C_T1': row['act_C_T1'],  # Time context 1
        'act_C_T2': row['act_C_T2'],  # Time context 2
        'act_C_T3': row['act_C_T3'],  # Time context 3
        'act_C_P1': row['act_C_P1'],  # Place context 1
        'act_C_P2': row['act_C_P2'],  # Place context 2
        'act_C_P3': row['act_C_P3']   # Place context 3
    }

# Maps actID to its context group, for 3, 4 and 5 groups
layer_mappings = {
    "M3": dict(zip(action_df['actID'], action_df['ContextG_act_3'].astype(int))),
    "M4": dict(zip(action_df['actID'], action_df['ContextG_act_4'].astype(int))),
    "M5": dict(zip(action_df['actID'], action_df['ContextG_act_5'].astype(int))),
}
layer_labels = {layer: sorted(set(mapping.values())) for layer, mapping in layer_mappings.items()}
for layer, labels in layer_labels.items():
    print(f"{layer} labels: {labels}")

# ========== SELECTED ASPECT ==========
# Only the 'activity' aspect is used
aspect_groups_lst = ['activity']

# User weight of the activity aspect (a_F2)
uID_activity_scores = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['a_F2']))

#%% STEP 4: SELECT ACTIONS BASED ON CHOSEN ASPECTS
##==================================================================================
print("========== STEP 4: SELECTING ACTIONS FOR GIVEN ASPECTS ==========")

# Action IDs whose question belongs to the selected aspect group(s)
single_actID_lst = [actID for group in aspect_groups_lst
                    for actID in actID_qID_dc if actID_qID_dc[actID] in group_qLst[group]]
print(f"Selected actions: {len(single_actID_lst)}")

#%% STEP 5: GENERATE COMBINATIONS OF ACTIONS
##==================================================================================
print("========== STEP 5: GENERATING COMBINATIONS OF ACTIONS ==========")

# All combinations of actIDs up to the maximum length
seq_actID_lst = erst.get_list_of_actions(single_actID_lst, act_max_len)

#%% STEP 6: LOAD OR BUILD D_lst DATA MATRIX
##==================================================================================
print("========== STEP 6: LOAD / BUILD D_lst ==========")

"""
D_lst = list of [user_id, action_combination, context, rating].
It captures how relevant an action combination is for a user and is the central object
for training and evaluating the recommender.

Rating rule (see erst.get_score_estimation and erst.get_dataMat):
- S[u, a] = a_F2[u] * answer[u, q(a)], where q(a) is the question of action a.
- The rating of a combination is the sum of S[u, a] * scaled_answer[u, q(a)] over its
  actions, minus the penalty 1 - (product of pairwise compatibilities).
- Only combinations with a rating above the threshold r_T are kept; the kept ratings are
  then rescaled to [r_T, 5].
"""

if USE_EXISTING_D_LST:
    with open(EXISTING_D_LST_PATH, "rb") as f:
        D_lst = pickle.load(f)
    print(f"Loaded existing D_lst: {EXISTING_D_LST_PATH} | rows={len(D_lst)}")

else:
    start_time = time.time()

    # Relevance score of each action for each user
    actID_score_df = erst.get_actID_score_df(
        uIDs, single_actID_lst, actID_qID_dc, uID_activity_scores, all_answers_df
    )

    # Compatibilities between the selected actions
    actID_compat_df = action_compat_df.loc[single_actID_lst, single_actID_lst]

    # User-action matrix of scaled answers to the question linked to each action
    uID_qID_answers_df = erst.get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst)
    uID_actID_answers_df = pd.DataFrame(index=uID_qID_answers_df.index)
    for actID in single_actID_lst:
        uID_actID_answers_df[actID] = uID_qID_answers_df[actID_qID_dc[actID]]

    D_lst = erst.get_dataMat(
        uIDs,
        seq_actID_lst,
        uID_actID_answers_df,
        actID_score_df,
        actID_compat_df,
        r_T,
        meth_code,
        actID_context_dc=actID_context_dc,
        normalize=True
    )
    print(f"\n DONE: D_lst generated with {len(D_lst)} entries in {time.time() - start_time:.2f} seconds.")

    D_lst_path.mkdir(parents=True, exist_ok=True)
    out_path = D_lst_path / f"D_lst_full_{timestamp}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(D_lst, f)
    print(f"Saved full D_lst to: {out_path}")

if test_mode:
    D_lst = D_lst[:TEST_D_LST_MAX_ROWS]
    print(f"[test_mode] Sliced D_lst to first {TEST_D_LST_MAX_ROWS} rows | rows={len(D_lst)}")

ratings = [x[3] for x in D_lst]

print("\n========== D_lst STATISTICS ==========")
print(f"Total rows in D_lst: {len(D_lst)}")
print(f"Unique users: {len(set(x[0] for x in D_lst))}")
print(f"Unique actions: {len(set(x[1] for x in D_lst))}")
print(f"Min rating: {min(ratings):.3f}")
print(f"Max rating: {max(ratings):.3f}")
print(f"Mean rating: {np.mean(ratings):.3f}")

# ========== CONTEXT ANALYSIS ==========
# Analyze distribution of time (C_T1) and place (C_P1) contexts
context_counts = Counter(
    (x[2][0].get("C_T1", "None"), x[2][0].get("C_P1", "None"))
    for x in D_lst if isinstance(x[2], tuple) and isinstance(x[2][0], dict)
)

print("\n Top 10 contexts (C_T1, C_P1):")
for ctx, count in context_counts.most_common(10):
    print(f"{ctx}: {count}x")

# The rating histogram of the thesis (figure 3.1) is drawn by presentation/make_slide_figures.py

#%% STEP 7: COMPARE ALGORITHMS (SVD, KNNBasic, BaselineOnly, NMF)
##==================================================================================
print("========== STEP 7: ALGORITHM COMPARISON ==========")

"""
This step prepares the rating data for Surprise and reserves the final test pairs.
Then, several collaborative filtering algorithms are evaluated using ShuffleSplit cross-validation.
"""

print("\n========== Preparing DataFrame for Surprise... ==========")
df = pd.DataFrame(D_lst, columns=['user_id', 'item_id', 'context', 'rating'])
# Remove 'context' because Surprise does not support it
df_for_surprise = df[['user_id', 'item_id', 'rating']]
print(f"Shape of df_for_surprise: {df_for_surprise.shape}")

# Prepare data
reader = Reader(rating_scale=(0, 5))
# Reserve final test pairs BEFORE algorithm comparison or hyperparameter search.
# CV and GridSearch below receive only the remaining development data.
development_idx, heldout_idx = next(
    ShuffleSplit(n_splits=1, test_size=test_size, random_state=42).split(df_for_surprise)
)
data = Dataset.load_from_df(df_for_surprise.iloc[development_idx], reader)
testset = list(df_for_surprise.iloc[heldout_idx].itertuples(index=False, name=None))

# Define algorithms and evaluate using CV
if not SKIP_TUNING:
    algorithms = {
        "SVD": SVD,
        "KNNBasic": KNNBasic,
        "BaselineOnly": BaselineOnly,
        "NMF": NMF
    }

    metrics_summary = []

    print("\nPerforming cross-validation for each algorithm...")
    for name, algo_class in tqdm(algorithms.items(), desc="Evaluating Algorithms"):
        print(f"\n=== Evaluating {name} ===")
        cv_df = erst.perform_cross_validation(
            data=data,
            model_class=algo_class,
            algorithm_name=name,
            cv_type='shuffle',
            n_splits=n_splits,
            test_size=test_size,
            random_state=42
        )

        # cv_df columns: Algorithm, Metric, Mean, Std
        mean_metrics = cv_df.set_index('Metric')['Mean']

        metrics_summary.append({
            'algorithm': name,
            'rmse': round(float(mean_metrics['RMSE']), 4),
            'mae': round(float(mean_metrics['MAE']), 4),
            'mse': round(float(mean_metrics['MSE']), 4),
            'fcp': round(float(mean_metrics['FCP']), 4)
        })

    # Convert to DataFrame and export
    summary_df = pd.DataFrame(metrics_summary)
    print("\nCross-validation results summary:")
    print(summary_df)

    summary_df.to_excel(evaluation_path / f"algorithm_comparison_{timestamp}.xlsx", index=False)

    # Save LaTeX table
    erst.save_df_as_latex_table(
        df=summary_df.rename(columns={"algorithm": "Algoritem", "rmse": "RMSE", "mae": "MAE", "mse": "MSE", "fcp": "FCP"}),
        out_dir=tabs_path,
        filename_stem=f"algorithm_comparison_{timestamp}",
        caption=f"Primerjava algoritmov na razvojnem delu podatkov (ShuffleSplit, {n_splits} ponovitev, {int(test_size*100)}\\% test).",
        label="tab:algorithm_comparison",
        float_format="{:.3f}",
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary_df["algorithm"], summary_df["rmse"], color="lightblue")
    ax.set_ylabel("Povprečni RMSE")
    ax.set_title("Primerjava algoritmov - povprečni RMSE")
    fig.tight_layout()
    fig.savefig(figs_path / f"algorithm_rmse_comparison_{timestamp}.png")
    plt.close(fig)

#%% STEP 8: HYPERPARAMETER TUNING (GRID SEARCH FOR SVD)
##==================================================================================
print("========== STEP 8: GRID SEARCH & SVD TUNING ==========")

"""
SVD model is tuned using GridSearch to find optimal hyperparameters.
"""

if not SKIP_TUNING:
    param_grid = {
        'random_state': [42],
        'n_factors': [50] if test_mode else [50, 100, 150],
        'n_epochs': [20] if test_mode else [20, 30],
        'lr_all': [0.002] if test_mode else [0.002, 0.005],
        'reg_all': [0.02] if test_mode else [0.02, 0.05]
    }

    # Run grid search to find optimal hyperparameters
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'],
                      cv=KFold(n_splits=n_splits, random_state=42, shuffle=True))
    gs.fit(data)

    best_params = gs.best_params['rmse']
    print(f"Best params: {best_params}")
    print(f"Best RMSE score: {gs.best_score['rmse']}")
else:
    best_params = BEST_PARAMS

#%% STEP 9: CROSS-VALIDATION FOR TUNED SVD
##==================================================================================
print("========== STEP 9: CROSS-VALIDATION (TUNED SVD) ==========")

"""
This step validates the performance of the tuned SVD model using shuffle split cross-validation.
"""

if not SKIP_TUNING:
    print(f"\nEvaluating tuned SVD model with {n_splits} CV...")
    cv_results_df = erst.perform_cross_validation(
        data=data,
        model_class=lambda: SVD(**best_params),
        algorithm_name='SVD_tuned',
        cv_type='shuffle',
        n_splits=n_splits,
        test_size=test_size,
        random_state=42
    )

    cv_results_df.to_excel(evaluation_path / f"cv_results_tuned_SVD_{timestamp}.xlsx", index=False)

    # Save LaTeX table
    erst.save_df_as_latex_table(
        df=cv_results_df.drop(columns="Algorithm").rename(columns={"Metric": "Metrika", "Mean": "Povprečje", "Std": "Standardni odklon"}),
        out_dir=tabs_path,
        filename_stem=f"cv_results_tuned_SVD_{timestamp}",
        caption=f"Razvojni rezultati po izbiri nastavitev (ShuffleSplit, {n_splits} ponovitev, {int(test_size*100)}\\% test) za prilagojeni model SVD.",
        label="tab:cv_results_tuned_svd",
        float_format="{:.4f}",
        index=False
    )

    print("\n Cross-validation results for tuned SVD:")
    print(cv_results_df)

#%% STEP 10: FINAL TRAINING AND MATRIX FACTORIZATION
##==================================================================================
print("========== STEP 10: FINAL TRAINING & EVALUATION ==========")

"""
Train on development data; evaluate on the untouched final test pairs.
"""

# Train on all development pairs, with the reserved final test set unchanged
trainset = data.build_full_trainset()

print("\n========== TRAIN(TEST)SET INFO ==========")
print(f"Number of users in trainset: {trainset.n_users}")
print(f"Number of items in trainset: {trainset.n_items}")
print(f"Trainset length: {trainset.n_ratings}")
print(f"Testset length: {len(testset)}")
print(f"Total unique actions in D_lst: {len(set(x[1] for x in D_lst))}")

# Initialize and train SVD model with best parameters
model = SVD(**best_params)
model.fit(trainset)

# Extract P and Q from the trained SVD; the entire activity combination is the factor key
user_factors = {trainset.to_raw_uid(uid): model.pu[uid] for uid in range(trainset.n_users)}
item_factors = {erst.normalize_item(trainset.to_raw_iid(iid)): model.qi[iid] for iid in range(trainset.n_items)}

# Predict ratings for the testset
predictions = model.test(testset)

print("\n========== MATRIX FACTORIZATION FACTORS ==========")
print(f"Shape of user factors: {model.pu.shape}")
print(f"Shape of item factors: {model.qi.shape}")

# Compute metrics on the test set
print("\n========== TEST SET EVALUATION ==========")
print(f"RMSE: {accuracy.rmse(predictions, verbose=False)}")
print(f"MAE:  {accuracy.mae(predictions, verbose=False)}")
print(f"MSE:  {accuracy.mse(predictions, verbose=False)}")
try:
    print(f"FCP:  {accuracy.fcp(predictions, verbose=False)}")
except ValueError:
    print("FCP:  N/A (premalo parov ocen na uporabnika v testsetu)")

#%% STEP 11: GENERATE RECOMMENDATIONS (4.1/4.2/4.3) AND EXPORT
##==================================================================================
print("========== STEP 11: GENERATE RECOMMENDATIONS (4.1/4.2/4.3) ==========")

# reproducibility (important if random)
np.random.seed(42)
python_random.seed(42)

# Pool only from the activities that can be recommended (34, not 106)
C_T_pool, C_P_pool = erst.build_context_pool({a: actID_context_dc[a] for a in single_actID_lst})
print(f"Context pool sizes: |C_T|={len(C_T_pool)} |C_P|={len(C_P_pool)}")

# Recommendations (uID, activity_tuple, score)
recs_41 = []  # baseline top 5
recs_42 = []  # context filtered
recs_layers = {layer: {g: [] for g in labels} for layer, labels in layer_labels.items()}
context_42_by_user = {}  # one context per user

# Rank only the held-out candidates for each user, using predictions already
# computed in STEP 10. Training pairs are never recommendation candidates here.
test_predictions_by_user = defaultdict(list)
for prediction in predictions:
    test_predictions_by_user[prediction.uid].append(
        (prediction.uid, erst.normalize_item(prediction.iid), float(prediction.est))
    )

for uID in tqdm(sorted(test_predictions_by_user), desc="Generating recs (held-out)"):

    context_42 = erst.sample_random_context(C_T_pool, C_P_pool) if USE_RANDOM_CONTEXT_42 else FIXED_CONTEXT_42
    context_42_by_user[uID] = context_42

    # Candidates in ranking order (unfiltered)
    ranked = sorted(test_predictions_by_user[uID], key=lambda rec: (-rec[2], rec[1]))

    # 4.1 Baseline: take top 5 directly
    recs_41.extend(ranked[:top_k_final_recommendations])

    # 4.2 Context filter: first 5 feasible in the user's context
    recs_42.extend(erst.first_k(ranked, top_k_final_recommendations,
                                lambda item: erst.item_context_feasible(item, context_42, actID_context_dc)))

    # 4.3 Grouped contexts: first 5 per group
    for layer, mapping in layer_mappings.items():
        for g in layer_labels[layer]:
            recs_layers[layer][g].extend(erst.first_k(ranked, top_k_final_recommendations,
                                                      lambda item: erst.item_in_group(item, g, mapping)))

# Popularity baseline: rank each user's held-out items by the item's mean training rating,
# without context and with the same context filter as 4.2
item_mean = {erst.normalize_item(trainset.to_raw_iid(i)): float(np.mean([r for _, r in trainset.ir[i]]))
             for i in trainset.all_items()}
recs_pop = []
recs_pop_42 = []
for uID in sorted(test_predictions_by_user):
    ranked_pop = sorted(((uID, item, item_mean.get(item, 0.0)) for _, item, _ in test_predictions_by_user[uID]),
                        key=lambda rec: (-rec[2], rec[1]))
    recs_pop.extend(ranked_pop[:top_k_final_recommendations])
    recs_pop_42.extend(erst.first_k(ranked_pop, top_k_final_recommendations,
                                    lambda item: erst.item_context_feasible(item, context_42_by_user[uID], actID_context_dc)))

# ========== SAVE EXPORTS ==========
def export_rows(recs, contexts=None, with_factors=False):
    """Recommendations as table rows, optionally with the used context and the SVD factors P/Q."""
    rows = []
    for uID, item, score in recs:
        row = {"uID": uID, "ActivitySequence": erst.serialize_item(item), "SequenceLength": len(item), "Score": round(score, 6)}
        if contexts is not None:
            row.update(contexts[uID])
        if with_factors:
            row["MF_P"] = user_factors.get(uID, np.zeros(model.n_factors)).tolist()
            row["MF_Q"] = item_factors.get(item, np.zeros(model.n_factors)).tolist()
        rows.append(row)
    return pd.DataFrame(rows)


exports = {
    "recommendations_4_1_with_PQ": export_rows(recs_41, with_factors=True),
    "recommendations_4_2_with_PQ": export_rows(recs_42, contexts=context_42_by_user, with_factors=True),
}
for layer, recs_by_group in recs_layers.items():
    for g, recs in recs_by_group.items():
        exports[f"recommendations_{layer}_group{g}"] = export_rows(recs)

for stem, export_df in exports.items():
    export_df.to_excel(recommendations_path / f"{stem}_{timestamp}.xlsx", index=False)
print(f"Saved {len(exports)} recommendation exports to: {recommendations_path}")

print("STEP 11 done. Containers ready for STEP 12 evaluation:")
print(f"recs_41: {len(recs_41)} | recs_42: {len(recs_42)}")
for layer, recs_by_group in recs_layers.items():
    print(f"{layer} groups: {[len(recs) for recs in recs_by_group.values()]}")

#%% STEP 12: EVALUATE (4.1/4.2/4.3) AND EXPORT RESULTS
##==================================================================================
print("========== STEP 12: EVALUATE (4.1/4.2/4.3) ==========")

# Ground truth from reserved test pairs (complete activity items)
D_triplets = [(uid, erst.normalize_item(item), float(rating)) for uid, item, rating in testset]
# 4.2 reference restricted to the user's context (same rule as for candidates)
D_triplets_42 = [(uid, item, r) for uid, item, r in D_triplets
                 if erst.item_context_feasible(item, context_42_by_user[uid], actID_context_dc)]
n_test_users = len(test_predictions_by_user)

print(f"D_triplets (complete items) size: {len(D_triplets)}")
print(f"Eval params: top_n_groundtruth={top_n_groundtruth} | k_eval={top_k_final_recommendations}")


test_ratings = {(uid, item): rating for uid, item, rating in D_triplets}


def result_row(category, recs, reference):
    """One row of the results table: recommendations evaluated against a reference."""
    m = erst.evaluate_topk(reference, recs, top_k_final_recommendations, top_n_groundtruth)
    return {"Category": category, "Precision": m["P"], "Recall": m["R"], "F1": m["F1"], "F1_ties": m["F1_ties"],
            "Coverage": erst.coverage(recs, n_test_users),
            "AverageScore": erst.average_score(recs),
            "AverageReferenceRating": erst.average_reference_rating(recs, test_ratings)}


def random_row(category, candidates, reference):
    """Expected values of a random ranking with the same candidates and reference."""
    m = erst.expected_random_topk(reference, candidates, top_k_final_recommendations, top_n_groundtruth)
    return {"Category": category, "Precision": m["P"], "Recall": m["R"], "F1": m["F1"], "F1_ties": m["F1_ties"],
            "Coverage": m["n_covered"] / n_test_users,
            "AverageScore": "--",
            "AverageReferenceRating": m["avg_rating"]}


# Comparison rows use the same candidates, filter and reference as the SVD row above them
result_rows = [
    result_row("Brez konteksta (baseline)", recs_41, D_triplets),
    result_row("Popularnost (baseline)", recs_pop, D_triplets),
    random_row("Naključno (pričakovana vrednost)", D_triplets, D_triplets),
    result_row("S kontekstom, globalna referenca", recs_42, D_triplets),
    random_row("Naključno s kontekstom, globalna referenca", D_triplets_42, D_triplets),
    result_row("S kontekstom, kontekstna referenca", recs_42, D_triplets_42),
    result_row("Popularnost s kontekstom, kontekstna referenca", recs_pop_42, D_triplets_42),
    random_row("Naključno s kontekstom, kontekstna referenca", D_triplets_42, D_triplets_42),
]

# 4.3 Grouped metrics: average across groups, weighted by the number of users
for layer, mapping in layer_mappings.items():
    df_layer, layer_avg = erst.evaluate_groups(
        D_triplets, recs_layers[layer], mapping,
        top_k_final_recommendations, top_n_groundtruth, n_test_users, layer
    )
    erst.round_floats(df_layer).to_excel(recommendations_path / f"evaluation_metrics_{layer}_per_group_{timestamp}.xlsx", index=False)
    result_rows.append({"Category": f"{layer} (uteženo po uporabnikih)", "Coverage": "--", **layer_avg})

# ========== RESULTS TABLE ==========
results_df = erst.round_floats(pd.DataFrame(result_rows))

# The LaTeX table shows only F1, which is valid only if precision and recall equal F1 in every row
prf_equal = bool((results_df["Precision"] == results_df["Recall"]).all() and (results_df["Precision"] == results_df["F1"]).all())
if not prf_equal:
    print("OPOZORILO: natančnost, priklic in F1 se v vsaj eni vrstici razlikujejo; "
          "tabela LaTeX results_all_layers ne bo izvožena (xlsx vsebuje vse tri stolpce).")

out_table = recommendations_path / f"results_all_layers_{timestamp}.xlsx"
results_df.to_excel(out_table, index=False)

print("\n========== RESULTS (ALL LAYERS) ==========")
print(results_df.to_string())
print("Opomba: kontekstna referenca in M3/M4/M5 uporabljajo druge referenčne množice (M3/M4/M5 uteženo po uporabnikih); višja metrika sama ne dokazuje izboljšanja.")
print(f"Saved: {out_table}")

if prf_equal:
    erst.save_df_as_latex_table(
        df=results_df[["Category", "F1", "F1_ties", "Coverage", "AverageReferenceRating"]].replace(
            {"Category": {"referenca": "ref.", "uteženo po uporabnikih": "uteženo"}}, regex=True).rename(columns={
            "Category": "Različica", "F1_ties": "F1 z izen.", "Coverage": "Pokritost", "AverageReferenceRating": "Povp. ref. ocena"}),
        out_dir=tabs_path,
        filename_stem=f"results_all_layers_{timestamp}",
        caption="Rezultati na testnih parih. Natančnost in priklic sta pri vseh vrsticah enaka F1. "
                "F1 z izen. upošteva izenačenja pri meji referenčnega seznama, povp. ref. ocena je povprečna testna ocena vrnjenih priporočil. "
                "Kontekstna referenca in M3--M5 uporabljajo ožje reference in niso neposredno primerljivi z globalno referenco.",
        label="tab:results_all_layers",
        float_format="{:.3f}",
        index=False
    )

# ========== CONTEXT REFERENCE: USERS BY NUMBER OF FEASIBLE TEST ITEMS ==========
users_feasible_anywhere = {row[0] for row in D_lst if row[0] in context_42_by_user
                           and erst.item_context_feasible(row[1], context_42_by_user[row[0]], actID_context_dc)}
breakdown_df = erst.round_floats(erst.context_breakdown(
    D_triplets_42, {"SVD": recs_42, "Popularity": recs_pop_42},
    set(test_predictions_by_user), users_feasible_anywhere,
    top_k_final_recommendations, top_n_groundtruth
))
breakdown_df.to_excel(recommendations_path / f"context_breakdown_{timestamp}.xlsx", index=False)
print("\n========== CONTEXT REFERENCE BY NUMBER OF FEASIBLE TEST ITEMS ==========")
print(breakdown_df.to_string())

erst.save_df_as_latex_table(
    df=breakdown_df[["FeasibleTestItems", "NumUsers", "F1_SVD", "F1_Popularity", "F1_Random"]].fillna("--").replace(
        {"FeasibleTestItems": {"0, skladna kombinacija le v razvojnem delu": "0 (skladna le v razvojnem delu)",
                               "0, brez skladne kombinacije v celotnem naboru": "0 (brez skladne v celotnem naboru)"}}).rename(columns={
        "FeasibleTestItems": "Skladne testne kombinacije", "NumUsers": "Uporabniki",
        "F1_SVD": "SVD", "F1_Popularity": "Popularnost", "F1_Random": "Naključno"}),
    out_dir=tabs_path,
    filename_stem=f"context_breakdown_{timestamp}",
    caption="F1 s kontekstno referenco glede na število skladnih testnih kombinacij uporabnika. "
            "Naključno je pričakovana vrednost naključnega rangiranja skladnih kombinacij.",
    label="tab:context_breakdown",
    float_format="{:.3f}",
    index=False
)
print(f"Saved LaTeX tables to: {tabs_path}")

# Overview figure of the results table (strict F1, tie-aware F1, coverage)
erst.plot_results_overview(results_df, figs_path / f"results_overview_{timestamp}")
print(f"Saved results overview figure to: {figs_path}")

# %%
