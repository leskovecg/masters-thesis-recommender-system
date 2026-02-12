"""
EXPLAINABLE ELDERLY RECOMMENDER SYSTEM PIPELINE
===================================================

This script implements a modular and explainable recommender system for suggesting daily activities to elderly users.
It integrates multiple data sources (user responses, health dimensions, contextual metadata) and applies collaborative filtering 
(SVD, KNN, NMF, BaselineOnly) to build personalized and context-aware recommendations.

The main components of the pipeline are:

STEP 0: Import required libraries
STEP 1: Configuration & settings
STEP 2: Load user and activity data from Excel
STEP 3: Build dictionaries from context definitions
STEP 4: Select actions based on chosen aspects
STEP 5: Generate simplified context strings
STEP 6: Generate sequences of actions
STEP 7: Trim users/actions for testing (optional)
STEP 8: Compute action relevance scores
STEP 9: Load and filter the compatibility matrix
STEP 10: Build the user-action response matrix
STEP 11: Build or load D_lst data matrix
STEP 12: Compare algorithms using Surprise
STEP 13: Hyperparameter tuning (GridSearch for SVD)
STEP 14: Cross-validation for tuned SVD
STEP 15: Final model training and matrix factorization
STEP 16: Generate recommendations (4.1/4.2/4.3) and export
STEP 18: Evaluate recommendations with Precision / Recall / F1
"""

#%% STEP 0: IMPORT REQUIRED LIBRARIES
##==================================================================================
print("========== STEP 0: IMPORTING LIBRARIES ==========")

import os
from random import random
import numpy as np
import cProfile
import seaborn as sns
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic, BaselineOnly, NMF, accuracy
from sklearn.model_selection import ShuffleSplit
from surprise.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from collections import defaultdict
from pathlib import Path
import datetime
import pickle
import elderly_recsys_tools as erst
import importlib
importlib.reload(erst)  # Reload module to get latest changes
import time
from tqdm import tqdm
from collections import Counter

#%% STEP 1: CONFIGURATION & SETTINGS
##==================================================================================
print("========== STEP 1: GLOBAL SETTINGS ==========")

# ========== ENVIRONMENT & PATHS ==========
BASE_DIR = Path(__file__).resolve().parents[1]

data_path = BASE_DIR / 'data'
annotations_path = data_path / 'annotations'
context_path = data_path / 'context'
D_lst_path = data_path / 'D_lst'
questionnaires_path = data_path / 'questionnaires'
scores_path = data_path / 'scores'

outputs_path = BASE_DIR / 'outputs'
evaluation_path = outputs_path / 'evaluation'
tabs_path = outputs_path / 'latex' / 'tabs'  # LaTeX tables directory
figs_path = outputs_path / 'latex' / 'figs'
recommendations_path = outputs_path / 'recommendations'

# Generate timestamp for output versioning (YYYYMMDD_HHMMSS)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ========== EXECUTION MODES ==========
test_mode = False                # Enable test mode for faster debugging
USE_EXISTING_D_LST = True       # Load existing D_lst or generate new one
USE_RANDOM_CONTEXT_42 = True    # TRUE = random context, FALSE = fixed context_42
RANDOM_CONTEXT_MODE = "per_user"  # "per_user" or "per_rec"

# ========== DATA PROCESSING PARAMETERS ==========
act_max_len = 3                 # Maximum allowed length of action sequences
meth_code = 'score'             # Method for relevance computation
r_T = 0.3                       # Threshold for relevance inclusion
TEST_D_LST_MAX_ROWS = 100000    # Max rows to load when test_mode=True

# ========== MODEL & EVALUATION PARAMETERS ==========
M = 100                         # Number of latent features for SVD (will be updated after training)
n_splits = 10                   # Number of CV splits (use 2 for test_mode, 10 for production)
test_size = 0.25                # Proportion of dataset for test split
n_recommendations = 20          # Top candidates returned (before context filtering)
top_k_final_recommendations = 5 # Number of final context-aware recommendations
top_n_groundtruth = 5           # Top ground-truth items for evaluation

# ========== CONTEXT & RECOMMENDATION SETTINGS ==========
FIXED_CONTEXT_42 = {'C_T': 'dopoldne', 'C_P': 'doma'}  # Default context if not randomized

# ========== EXISTING D_LST PATH (relative to BASE_DIR) ==========
EXISTING_D_LST_PATH = D_lst_path / 'D_lst_full_20260202_150312.pkl'


#%% STEP 2: LOAD USER AND ACTIVITY DATA FROM EXCEL
##==================================================================================
print("========== STEP 2: LOADING DATA FROM EXCEL FILES ==========")

# Load activity data, replace -1 with NaN, and rename columns with 'Ac_' prefix
activities_df = pd.read_excel(questionnaires_path / 'Activities.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
activities_df.rename(columns={nm:'Ac_' + nm for nm in activities_df.columns}, inplace=True)

# Load mental health data, replace -1 with NaN, and rename columns with 'Mh_' prefix
mentalHealth_df = pd.read_excel(questionnaires_path / 'MentalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
mentalHealth_df.rename(columns={nm:'Mh_' + nm for nm in mentalHealth_df.columns}, inplace=True)

# Load physical health data, replace -1 with NaN, and rename columns with 'Ph_' prefix
physicalHealth_df = pd.read_excel(questionnaires_path / 'PhysicalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
physicalHealth_df.rename(columns={nm:'Ph_' + nm for nm in physicalHealth_df.columns}, inplace=True)

# Load social health data, replace -1 with NaN, and rename columns with 'Sh_' prefix
socialHealth_df = pd.read_excel(questionnaires_path / 'SocialHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
socialHealth_df.rename(columns={nm:'Sh_' + nm for nm in socialHealth_df.columns}, inplace=True)

# Load context definitions for activities (time and place conditions)
activityContextGen_df = pd.read_excel(context_path / 'ActivityContextGen_v12.xlsx', sheet_name='ActionLst').replace(-1, np.nan)

# Load user-specific scores and weights (how suitable each activity is for each user)
scores_and_wgt_df = pd.read_excel(scores_path / 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')

# Optional: other versions of the score data (commented out for now)
#scores_and_wgt_2_df = pd.read_excel(data_path + 'ml_data_with_scores_and_wgt_3_values.xlsx', sheet_name='scores_and_wgt_3_v', header=1)
#wgt_results_annotations_df = pd.read_excel(data_path + 'wgt_results_annotations_3009_4users.xlsx', sheet_name='wgt_results_annotations_3009_4u', header=1).drop(columns=['Unnamed: 0', 'Column1'])

# Merge all user responses (activities + mental + physical + social) into a single DataFrame
all_answers_df = activities_df.join(mentalHealth_df).join(physicalHealth_df, rsuffix='_r').join(socialHealth_df, rsuffix='_r')

# Load handcrafted compatibility matrix (with continuous values between 0 and 1)
action_compat_df = pd.read_excel(context_path / 'ActivityContextGen_v12.xlsx', sheet_name='ActionCompatibilities', index_col=0)

action_df = pd.read_excel(context_path / "ActivityContextGen_v12.xlsx", sheet_name="ActionLst")

#%% STEP 3: BUILD DICTIONARIES FROM CONTEXT DEFINITIONS
##==================================================================================
print("========== STEP 3: BUILDING DICTIONARIES & STRUCTURES ==========")

# Extract user IDs from the scores DataFrame
uIDs = list(np.sort(scores_and_wgt_df.index))

# Define lists of question IDs for different health/activity aspects
# These group questionnaire columns relevant to each domain
# These lists group specific columns (questions) related to activities, physical health, mental health, and social health. Each list contains column names that are relevant to the respective group.
activity_qs = ['Ac_AB4_1', 'Ac_AB4_2', 'Ac_AB4_3', 'Ac_AB4_4', 'Ac_AB4_5', 'Ac_AB4_8']
phy_health_qs = ['Ph_AB1_7', 'Ph_AB1_11', 'Ph_AB3', 'Ph_AB6_1', 'Ph_AB6_5', 'Ph_AB7_1', 'Ph_AB7_5', 'Ph_AB4_2', 'Ph_AB4_3', 'Ph_AB4_4']
ment_health_qs = ['Mh_A75_2', 'Mh_A75_3', 'Mh_A75_4', 'Mh_A75_5', 'Mh_AB1_14', 'Mh_A82_r1', 'Mh_A82_r3']
soc_health_qs = ['Sh_A83_r', 'Sh_sh_AB98_da_ne']

# This dictionary groups the question lists into a single structure, allowing easy access to each group by its category name.
group_qLst = {'activity': activity_qs,
              'phy_health': phy_health_qs,
              'ment_health': ment_health_qs,
              'soc_health': soc_health_qs}

# This dictionary maps different health and activity groups to specific factors (like F1, F2, etc.). These factors likely represent different components or categories relevant for the analysis.
group_factor_dc = {'Activities': 'F2', 
                   'PhysicalHealth-organskiSistemi': 'F1',
                   'PhysicalHealth-nacinZivljenja': 'F1',
                   'MentalHealth-osnovno': 'F1',
                   'MentalHealth-visje': 'F3'}

# Create dictionaries to both ways -  This helps in easily finding corresponding actions based on their IDs and vice versa.
actID_singleAct_dc = dict(zip(activityContextGen_df['actID'], activityContextGen_df['Single_action']))
singleAct_actID_dc = dict(zip(activityContextGen_df['Single_action'], activityContextGen_df['actID']))

# ========== INITIALIZE DICTIONARIES FOR ACTION MAPPINGS ==========
# These dictionaries store mappings between question IDs, actions, and their properties:
curr_qID = np.nan
curr_qIDs = []
qID_qtxt_dc = {} # Maps qID to question text
qID_singleAct_dc = {} # Maps qID to list of single actions
qID_actID_dc = {} # Maps qID to list of actIDs
actID_context_dc = {} # Maps actID to its context (time/place)
actID_props_dc = {} # Maps actID to its properties
qID_Group_dc = {} # Maps qID to its group/category
singleAct_qID_dc = {} # Maps unique single action to qID
actID_qID_dc = {} # Maps actID to qID

# Populate the above dictionaries using the context definition table
for ind, row in activityContextGen_df.iterrows():
    
    # Skip certain question IDs that are not covered in the analysis
    if row['qID'] not in ['A82_r1', 'A82_r3', 'A83_r']:
        if pd.notnull(row['qID']):
            # Process each new question ID only once
            if row['qID'] not in curr_qIDs:
                curr_qID = row['qID']
                curr_qIDs.append(curr_qID)
                qID_qtxt_dc[curr_qID] = row['qText']
                qID_Group_dc[curr_qID] = row['Group']
                qID_singleAct_dc[curr_qID] = []
                qID_actID_dc[curr_qID] = []

        last_qID = curr_qIDs[-1]

        # Link actions and action IDs to the current question
        single_act = row['Single_action']
        actID = row['actID']
        qID_singleAct_dc[curr_qID].append(single_act)
        #singleAct_qID_dc[curr_qID+'_'+str(single_act)] = curr_qID
        singleAct_qID_dc.update({curr_qID+'_'+str(single_act):curr_qID})
        qID_actID_dc[curr_qID].append(actID)
        actID_qID_dc.update({str(actID):curr_qID})

        # Store action properties (usefulness, difficulty, etc.)
        actID_props_dc[actID] = {
            'qID': last_qID,
            'action_prop_1': row['action_prop_1'],
            'action_prop_2': row['action_prop_2'],
            'action_prop_3': row['action_prop_3']
        }

        # Store action context (time and place descriptors)
        actID_context_dc[actID] = {
            'qID': last_qID,
            'act_C_T1': row['act_C_T1'],  # Time context 1
            'act_C_T2': row['act_C_T2'],  # Time context 2
            'act_C_T3': row['act_C_T3'],  # Time context 3
            'act_C_P1': row['act_C_P1'],  # Place context 1
            'act_C_P2': row['act_C_P2'],  # Place context 2
            'act_C_P3': row['act_C_P3']   # Place context 3
        }

# ========== CONVERT USER SCORES INTO DICTIONARIES ==========
# Create dictionaries mapping user IDs to their scores for each aspect
uID_activity_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['a_F2']))
uID_menHealOsn_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_osnovno_F1']))
uID_menHealVisje2_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F2']))
uID_menHealVisje4_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F4']))
uID_phyHealNacin_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_nacinZivljenja_F1']))
uID_phyHealOrganski_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_organskiSistemi_F1']))

# ========== SELECT ASPECT GROUPS ==========
# Available options: 'activity', 'phy_health', 'ment_health', 'soc_health'
# Currently using only 'activity' for simplicity
aspect_groups_lst = ['activity']  # Modify to include more aspects if needed

# Aggregate all score dictionaries by aspect/category

#score_groups_lst = ['activity', 'menHealOsn', 'menHealVisje2', 'menHealVisje4', 'phyHealNacin', 'phyHealOrganski']
uID_scores_dc = {
    'activity': uID_activity_scores_dc,
    'menHealOsn': uID_menHealOsn_scores_dc,
    'menHealVisje2': uID_menHealVisje2_scores_dc,
    'menHealVisje4': uID_menHealVisje4_scores_dc,
    'phyHealNacin': uID_phyHealNacin_scores_dc,
    'phyHealOrganski': uID_phyHealOrganski_scores_dc
}

#%% STEP 4: SELECT ACTIONS BASED ON CHOSEN ASPECTS
##==================================================================================
print("========== STEP 4: SELECTING ACTIONS FOR GIVEN ASPECTS ==========")

"""
We filter actions and actIDs based on the selected aspect group(s),
e.g. only actions related to the 'activity' aspect.
"""

# Generate a list of single actions for the selected aspect group(s)
single_act_lst = []
for group in aspect_groups_lst:
    for act in singleAct_qID_dc:
        for qID in group_qLst[group]:
            if qID in act:
                single_act_lst.append(act)

# Generate a list of action IDs for the selected aspect group(s)
single_actID_lst = []
for group in aspect_groups_lst:
    for actID in actID_qID_dc:
        for qID in group_qLst[group]:
            if qID in actID:
                single_actID_lst.append(actID)

#%% STEP 5: GENERATE SIMPLIFIED CONTEXT STRINGS
##==================================================================================
print("========== STEP 5: PROCESS CONTEXT STRINGS ==========")

"""
Generates a simplified 'kontekst' string from C_T1–C_T3 columns and trims the last invalid rows.
"""

# Generate simplified context strings by combining time context values
activityContextGen_df['kontekst'] = activityContextGen_df.apply(
    lambda row: erst.get_context(row['act_C_T1'], row['act_C_T2'], row['act_C_T3']), axis=1
)

# Extract context strings into a list
context_lst = activityContextGen_df['kontekst'].tolist()

# Remove last 6 elements (incomplete/placeholder entries)
context_lst = context_lst[:-6]

#%% STEP 6: GENERATE SEQUENCES OF ACTIONS
##==================================================================================
print("========== STEP 6: GENERATING SEQUENCES OF ACTIONS ==========")

"""
Creates all valid combinations of actions up to max length.
"""

# Generate all possible sequences of actions (represented by actIDs)
seq_actID_lst = erst.get_list_of_actions(single_actID_lst, act_max_len)

#%% STEP 7: TRIM USERS/ACTIONS FOR TESTING (OPTIONAL)
##==================================================================================
print("========== STEP 7: TRIM USERS/ACTIONS FOR TESTING ==========")

"""
For faster evaluation, we limit how many users/actions we consider.
"""

# Use full dataset of users and actions (can be reduced for faster testing)
uIDs_n, acts_n = len(uIDs), len(seq_actID_lst)

# Select subset of users and action sequences
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n]

# Extract only single-action sequences for relevance computation.
actID_lstIn = [act[0] for act in seq_actID_lstIn if len(act)==1]

#%% STEP 8: COMPUTE ACTION RELEVANCE SCORES
##==================================================================================
print("========== STEP 8: COMPUTING ACTION RELEVANCE SCORES ==========")

"""
Calculates how relevant each action is for each user based on scores and answers.
"""

# Compute action relevance scores based on user responses and weights
actID_score_df = erst.get_actID_score_df(
    uIDsIn, actID_lstIn, actID_qID_dc, 
    uID_scores_dc, all_answers_df, 
    aspect_groups_lst, meth_code
    )

# Note: Compatibility matrix is loaded from Excel (not computed here)

#%% STEP 9: LOAD AND FILTER THE COMPATIBILITY MATRIX
##==================================================================================
print("========== STEP 9: LOAD AND FILTER COMPATIBILITY MATRIX ==========")

"""
We take the full handcrafted matrix and filter to keep only actions we're working with.
"""

# Filter handcrafted matrix to include only relevant actions
actID_compat_df = action_compat_df.loc[actID_lstIn, actID_lstIn]

#%% STEP 10: BUILD THE USER-ACTION RESPONSE MATRIX
##==================================================================================
print("========== STEP 10: BUILDING USER-ACTION RESPONSE MATRIX ==========")

"""
This builds a matrix where each cell is user's answer to the question linked to an action.
"""

# Get user responses to questions for selected aspects
uID_qID_answers_df = erst.get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst)

# Build user-action response matrix
# Rows = users, Columns = actions
# Each cell contains the user's answer to the question linked to that action
uID_actID_answers_df = pd.DataFrame(index=uID_qID_answers_df.index)
for actID in actID_lstIn:
    if actID_qID_dc[actID] in uID_qID_answers_df:
        uID_actID_answers_df[actID] = uID_qID_answers_df[actID_qID_dc[actID]]
    else:
        print ('Error:' + actID)

# ========== VISUALIZE USER-ACTION RELEVANCE HEATMAP ==========
# Rows = users, columns = actions
# Each cell shows relevance of action to user based on questionnaire responses
sns.heatmap(uID_actID_answers_df)
plt.title("Relevantnost akcij za posameznega uporabnika")
plt.xlabel("Akcije")
plt.ylabel("Uporabniki")
plt.tight_layout()
plt.savefig(figs_path / f"user_action_relevance_heatmap_{timestamp}.png")
plt.show()

#%% STEP 11: BUILD OR LOAD D_lst DATA MATRIX
##==================================================================================
print("========== STEP 11: BUILD / LOAD D_lst ==========")

"""
D_lst = list of tuples: (user_id, action_seq, context, relevance_score)
This matrix is the central object for training and evaluating the recommender

This matrix captures how relevant a given action sequence is for a particular user.

Scoring logic and assumptions:
- Each action is linked to a question (e.g., via actID → qID).
- A user's *answer* to that question indicates their affinity for the action.
- A user's *score* for the corresponding aspect group indicates overall importance.
- Relevance is calculated as:
      relevance = score * answer
- If the sequence includes multiple actions, pairwise compatibility scores between them are also considered.
- Only action sequences with relevance above a threshold `r_T` are kept in the final matrix.

Input components:
- `uID_actID_answers_df`: users’ responses to individual questions
- `actID_score_df`: importance scores per user for each action
- `actID_compat_df`: (optional) compatibility between action pairs

The result is used for generating and evaluating personalized activity recommendations.
"""


"""
1. Generate data matrix
2. Use Matrix factorisation to obtain full D. 
3. Define test uIDs
4. For uID in uIDs:
    - generate (select) contexts for uID
    - generate recommendation: argmax(D(uID, :))
    - filter those compatible with the context
    - recommend top-3 in parametric and textual form
"""

if USE_EXISTING_D_LST:
    if not os.path.exists(EXISTING_D_LST_PATH):
        raise FileNotFoundError(f"Missing EXISTING_D_LST_PATH: {EXISTING_D_LST_PATH}")

    with open(EXISTING_D_LST_PATH, "rb") as f:
        D_lst = pickle.load(f)

    # Store filename for reference
    d_lst_filename = Path(EXISTING_D_LST_PATH).name

    print(f"Loaded existing D_lst: {EXISTING_D_LST_PATH} | rows={len(D_lst)}")

    # ✅ FAST slice for test_mode
    if test_mode:
        D_lst = D_lst[:TEST_D_LST_MAX_ROWS]
        print(f"[test_mode] Sliced D_lst to first {TEST_D_LST_MAX_ROWS} rows | rows={len(D_lst)}")

else:
    start_time = time.time()

    # # Only sequences with a relevance score above the defined threshold (r_T) are included.
    # ZGRADI PODATKOVNO MATRICO Z UPORABO ROČNO VNESENE KOMPATIBILNOSTI
    D_lst = erst.get_dataMat(
                            uIDsIn, 
                            seq_actID_lstIn, 
                            uID_actID_answers_df, 
                            actID_score_df, 
                            actID_compat_df, 
                            r_T, 
                            meth_code, 
                            actID_context_dc=actID_context_dc, 
                            normalize=True  
                            )

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n DONE: D_lst generated with {len(D_lst)} entries in {duration:.2f} seconds.")

    d_lst_filename = f"D_lst_full_{timestamp}.pkl"
    out_path = data_path / 'D_lst'
    # SAVE .pkl:
    with open(out_path / d_lst_filename, "wb") as f:
        pickle.dump(D_lst, f)
    print(f"Saved full D_lst to: {out_path / d_lst_filename}")

    # ✅ optional: also slice in test_mode even for freshly generated
    if test_mode:
        D_lst = D_lst[:TEST_D_LST_MAX_ROWS]
        print(f"[test_mode] Sliced freshly-generated D_lst to first {TEST_D_LST_MAX_ROWS} rows | rows={len(D_lst)}")

ratings = [x[3] for x in D_lst]

print("\n========== D_lst STATISTICS ==========")
print(f"Total rows in D_lst: {len(D_lst)}")
print(f"Unique users: {len(set(x[0] for x in D_lst))}")
print(f"Unique actions: {len(set(x[1] for x in D_lst))}")
print(f"Min rating: {min(ratings):.3f}")
print(f"Max rating: {max(ratings):.3f}")
print(f"Mean rating: {np.mean(ratings):.3f}")

# Najvišje in najnižje ocene
sorted_by_rating = sorted(D_lst, key=lambda x: x[3])
print("\n Lowest ratings:")
for x in sorted_by_rating[:3]:
    print(x)

print("\n Highest ratings:")
for x in sorted_by_rating[-3:]:
    print(x)

# ========== CONTEXT ANALYSIS ==========
# Analyze distribution of time (C_T1) and place (C_P1) contexts
context_counts = Counter(
    (x[2][0].get("C_T1", "None"), x[2][0].get("C_P1", "None"))
    for x in D_lst if isinstance(x[2], tuple) and isinstance(x[2][0], dict)
)

print("\n Top 10 contexts (C_T1, C_P1):")
for ctx, count in context_counts.most_common(10):
    print(f"{ctx}: {count}x")

# ========== VISUALIZE RATINGS DISTRIBUTION ==========
plt.hist(ratings, bins=20)
plt.title("Porazdelitev ocen v podatkovni matriki")
plt.xlabel("Ocena")
plt.ylabel("Število vnosov")
plt.grid(True)
plt.savefig(figs_path / f"ratings_distribution_hist_{timestamp}.png")
plt.show()

#%% STEP 12: COMPARE ALGORITHMS (SVD, KNNBasic, BaselineOnly, NMF)
##==================================================================================
print("========== STEP 12: ALGORITHM COMPARISON ==========")

"""
This step loads a sample of D_lst and prepares the rating data for Surprise.
Then, several collaborative filtering algorithms are evaluated using ShuffleSplit cross-validation.
"""

print("\n========== Preparing DataFrame for Surprise... ==========")
df = pd.DataFrame(D_lst, columns=['user_id', 'item_id', 'context', 'rating'])
# Remove 'context' because Surprise does not support it
df_for_surprise = df[['user_id', 'item_id', 'rating']]
print(f"Shape of df_for_surprise: {df_for_surprise.shape}")

# Prepare data
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df_for_surprise, reader)

# Define algorithms and evaluate using CV
# ==========================================================

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

# Save results and plot
# ==========================================================

# Convert to DataFrame and export
summary_df = pd.DataFrame(metrics_summary)
print("\nCross-validation results summary:")
print(summary_df)

summary_df.to_excel(evaluation_path / f"algorithm_comparison_{timestamp}.xlsx", index=False)

# Save LaTeX table
erst.save_df_as_latex_table(
    df=summary_df,
    out_dir=tabs_path,
    filename_stem=f"algorithm_comparison_{timestamp}",
    caption=f"Primerjava algoritmov (ShuffleSplit, {n_splits} ponovitev, {int(test_size*100)}\\% test).",
    label="tab:algorithm_comparison",
    float_format="{:.3f}",
    index=False
)

plt.figure(figsize=(8,5))
plt.bar(summary_df["algorithm"], summary_df["rmse"], color="lightblue")
plt.ylabel("Povprečni RMSE")
plt.title("Primerjava algoritmov - povprečni RMSE")
plt.tight_layout()
plt.savefig(figs_path / f"algorithm_rmse_comparison_{timestamp}.png")
plt.show()

#%% STEP 13: HYPERPARAMETER TUNING (GRID SEARCH FOR SVD)
##==================================================================================
print("========== STEP 13: GRID SEARCH & SVD TUNING ==========")

"""
SVD model is tuned using GridSearch to find optimal hyperparameters.
"""

print("\n========== GRID SEARCH ==========")
# # Define hyperparameter search space for the SVD algorithm (from Surprise library)

param_grid = {
    'n_factors': [50] if test_mode else [50, 100, 150],
    'n_epochs': [20] if test_mode else [20, 30],
    'lr_all': [0.002] if test_mode else [0.002, 0.005],
    'reg_all': [0.02] if test_mode else [0.02, 0.05]
}

profile = cProfile.Profile() 
profile.enable() 

# Run grid search to find optimal hyperparameters
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=n_splits)
gs.fit(data)

best_params = gs.best_params['rmse']
print(f"Best params: {best_params}")
print(f"Best RMSE score: {gs.best_score['rmse']}")

profile.disable()

#%% STEP 14: CROSS-VALIDATION FOR TUNED SVD
##==================================================================================
print("========== STEP 14: CROSS-VALIDATION (TUNED SVD) ==========")

"""
This step validates the performance of the tuned SVD model using shuffle split cross-validation.
"""

# Run cross-validation with tuned SVD parameters
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
    df=cv_results_df,
    out_dir=tabs_path,
    filename_stem=f"cv_results_tuned_SVD_{timestamp}",
    caption=f"Povprečni rezultati (ShuffleSplit, {n_splits} ponovitev, {int(test_size*100)}\\% test) za prilagojeni model SVD.",
    label="tab:cv_results_tuned_svd",
    float_format="{:.3f}",
    index=False
)

# === Extract and display average metrics ===
print("\n Cross-validation results for tuned SVD:")
print(cv_results_df)

# Display cross-validation results
mean_metrics = cv_results_df.set_index('Metric')['Mean']

#%% STEP 15: FINAL TRAINING AND MATRIX FACTORIZATION
##==================================================================================
print("========== STEP 15: FINAL TRAINING & EVALUATION ==========")

"""
Train the final SVD model on all data and evaluate its performance.
"""

# Train the model on the training set (with test set for validation)
trainset, testset = train_test_split(data, test_size=test_size)

print("\n========== TRAIN(TEST)SET INFO ==========")
print(f"Number of users in trainset: {trainset.n_users}")
print(f"Number of items in trainset: {trainset.n_items}")
# print(f"Trainset length: {len(trainset)}")
print(f"Trainset length: {trainset.n_ratings}")

print(f"Testset length: {len(testset)}")
print(f"Total unique actions in D_lst: {len(set(x[1] for x in D_lst))}")


# Initialize and train SVD model with best parameters
model = SVD(**best_params)
model.fit(trainset)



# =============================================================================
# EXTRACT P and Q from trained SVD
# =============================================================================
user_factors = {trainset.to_raw_uid(uid): model.pu[uid] for uid in range(trainset.n_users)}
item_factors = {trainset.to_raw_iid(iid): model.qi[iid] for iid in range(trainset.n_items)}

# Normalize item keys (tuple -> first element)
item_factors_norm = {k[0] if isinstance(k, tuple) else k: v for k, v in item_factors.items()}

print(f"User factors: {len(user_factors)} | Item factors: {len(item_factors_norm)}")

# Predict ratings for the testset
predictions = model.test(testset)

print("\n========== MATRIX FACTORIZATION FACTORS ==========")
print(f"Shape of user factors: {model.pu.shape}")
print(f"Shape of item factors: {model.qi.shape}")

# Compute metrics on the test set
print("\n========== TEST SET EVALUATION ==========")
print(f"RMSE: {accuracy.rmse(predictions)}")
print(f"MAE:  {accuracy.mae(predictions)}")
print(f"MSE:  {accuracy.mse(predictions)}")
try:
    print(f"FCP:  {accuracy.fcp(predictions)}")
except ValueError:
    print("FCP:  N/A (premalo parov ocen na uporabnika v testsetu)")

#%% STEP 16: GENERATE RECOMMENDATIONS (4.1/4.2/4.3) AND EXPORT
print("========== STEP 16: GENERATE RECOMMENDATIONS (4.1/4.2/4.3) ==========")

# reproducibility (important if random)
np.random.seed(42)

# Update M to match trained SVD dimensions
M = int(model.pu.shape[1])


# Use context group checking function
_is_in_group = getattr(erst, "is_action_in_context_group", erst.is_action_in_context_group_local)

# =============================================================================
# EXTRACT CONTEXT GROUPS FROM action_df (already loaded in STEP 2)
# =============================================================================
group_mapping_cols = {"ContextG_act_3": "g3", "ContextG_act_4": "g4", "ContextG_act_5": "g5"}
seg_df = action_df[["actID"] + list(group_mapping_cols.keys())].copy()
seg_df = seg_df.dropna(subset=["actID"])
seg_df["actID"] = seg_df["actID"].astype(str)

# Convert group columns to numeric
for col in group_mapping_cols.keys():
    seg_df[col] = pd.to_numeric(seg_df[col], errors="coerce")

# Build group mappings
actID_to_g3 = {r.actID: int(r.ContextG_act_3) for r in seg_df.dropna(subset=["ContextG_act_3"]).itertuples(index=False)}
actID_to_g4 = {r.actID: int(r.ContextG_act_4) for r in seg_df.dropna(subset=["ContextG_act_4"]).itertuples(index=False)}
actID_to_g5 = {r.actID: int(r.ContextG_act_5) for r in seg_df.dropna(subset=["ContextG_act_5"]).itertuples(index=False)}

G3_labels = sorted(set(actID_to_g3.values()))
G4_labels = sorted(set(actID_to_g4.values()))
G5_labels = sorted(set(actID_to_g5.values()))

print(f"M3 labels: {G3_labels}")
print(f"M4 labels: {G4_labels}")
print(f"M5 labels: {G5_labels}")

# =============================================================================
# CONTEXT POOL (for 4.2 random contexts)
# =============================================================================
C_T_pool, C_P_pool = erst.build_context_pool(actID_context_dc)
print(f"Context pool sizes: |C_T|={len(C_T_pool)} |C_P|={len(C_P_pool)}")

# =============================================================================
# OUTPUT CONTAINERS (for STEP 18 eval)
# =============================================================================
recs_41 = []  # (uID, (actID,), score) baseline top5
recs_42 = []  # (uID, (actID,), score) context filtered

recs_M3 = {g: [] for g in G3_labels}
recs_M4 = {g: [] for g in G4_labels}
recs_M5 = {g: [] for g in G5_labels}

# =============================================================================
# EXPORT TABLES (for thesis)
# =============================================================================
rows_41 = []  # with P/Q
rows_42 = []  # with P/Q + used context

# Optional exports for 4.3 (no P/Q to keep size manageable)
rows_M3 = {g: [] for g in G3_labels}
rows_M4 = {g: [] for g in G4_labels}
rows_M5 = {g: [] for g in G5_labels}

# =============================================================================
# MAIN LOOP: one pass per user
# =============================================================================
for uID in tqdm(uIDsIn, desc="Generating recs (one pass)"):

    # Choose context for 4.2
    if USE_RANDOM_CONTEXT_42 and RANDOM_CONTEXT_MODE == "per_user":
        context_42 = erst.sample_random_context(C_T_pool, C_P_pool)
    else:
        context_42 = FIXED_CONTEXT_42

    # MF user vector P
    MF_P = user_factors.get(uID, np.zeros(M))

    # 1) Get top-N candidates once (unfiltered)
    topN = erst.get_recommendations(
        uID=uID,
        trainset=trainset,
        model=model,
        n_recommendations=n_recommendations
    )

    topN_norm = [(uid, erst.normalize_act_id(item_id), float(score)) for uid, item_id, score in topN]

    # -------------------------------------------------------------------------
    # 4.1 Baseline: take top5 directly
    # -------------------------------------------------------------------------
    top5_41 = topN_norm[:top_k_final_recommendations]
    recs_41.extend([(uID, (act_id,), score) for _, act_id, score in top5_41])

    for _, act_id, score in top5_41:
        MF_Q = item_factors_norm.get(act_id, np.zeros(M))
        rows_41.append({
            "uID": uID,
            "ActID": act_id,
            "Score": round(score, 6),
            "MF_P": MF_P.tolist() if hasattr(MF_P, "tolist") else list(MF_P),
            "MF_Q": MF_Q.tolist() if hasattr(MF_Q, "tolist") else list(MF_Q),
        })

    # -------------------------------------------------------------------------
    # 4.2 Context filter: filter topN -> first top5 feasible
    # supports per_rec random context
    # -------------------------------------------------------------------------
    filtered_42 = []
    for _, act_id, score in topN_norm:

        if USE_RANDOM_CONTEXT_42 and RANDOM_CONTEXT_MODE == "per_rec":
            context_42 = erst.sample_random_context(C_T_pool, C_P_pool)

        if erst.is_action_context_feasibleQ(act_id, context_42, actID_context_dc):
            filtered_42.append((uID, act_id, score, context_42))

        if len(filtered_42) >= top_k_final_recommendations:
            break

    recs_42.extend([(uID, (act_id,), score) for _, act_id, score, _ in filtered_42])

    for _, act_id, score, ctx in filtered_42:
        MF_Q = item_factors_norm.get(act_id, np.zeros(M))
        rows_42.append({
            "uID": uID,
            "ActID": act_id,
            "Score": round(score, 6),
            "C_T": ctx.get("C_T", ""),
            "C_P": ctx.get("C_P", ""),
            "MF_P": MF_P.tolist() if hasattr(MF_P, "tolist") else list(MF_P),
            "MF_Q": MF_Q.tolist() if hasattr(MF_Q, "tolist") else list(MF_Q),
        })

    # -------------------------------------------------------------------------
    # 4.3 Grouped contexts: filter topN by group id
    # take first top5 per group
    # -------------------------------------------------------------------------
    for g in G3_labels:
        out = []
        for _, act_id, score in topN_norm:
            if _is_in_group(act_id, g, actID_to_g3):
                out.append((uID, act_id, score))
            if len(out) >= top_k_final_recommendations:
                break
        recs_M3[g].extend([(uID, (act_id,), score) for _, act_id, score in out])
        for _, act_id, score in out:
            rows_M3[g].append({"uID": uID, "ActID": act_id, "Score": round(score, 6)})

    for g in G4_labels:
        out = []
        for _, act_id, score in topN_norm:
            if _is_in_group(act_id, g, actID_to_g4):
                out.append((uID, act_id, score))
            if len(out) >= top_k_final_recommendations:
                break
        recs_M4[g].extend([(uID, (act_id,), score) for _, act_id, score in out])
        for _, act_id, score in out:
            rows_M4[g].append({"uID": uID, "ActID": act_id, "Score": round(score, 6)})

    for g in G5_labels:
        out = []
        for _, act_id, score in topN_norm:
            if _is_in_group(act_id, g, actID_to_g5):
                out.append((uID, act_id, score))
            if len(out) >= top_k_final_recommendations:
                break
        recs_M5[g].extend([(uID, (act_id,), score) for _, act_id, score in out])
        for _, act_id, score in out:
            rows_M5[g].append({"uID": uID, "ActID": act_id, "Score": round(score, 6)})

# =============================================================================
# SAVE EXPORTS
# =============================================================================
# 4.1 + 4.2: thesis-ready with P/Q
df_41 = pd.DataFrame(rows_41)
df_42 = pd.DataFrame(rows_42)

out_41 = recommendations_path / f"recommendations_4_1_with_PQ_{timestamp}.xlsx"
out_42 = recommendations_path / f"recommendations_4_2_with_PQ_{timestamp}.xlsx"

df_41.to_excel(out_41, index=False)
df_42.to_excel(out_42, index=False)

print(f"Saved: {out_41}")
print(f"Saved: {out_42}")

# 4.3: per group exports (optional but useful for appendix / debugging)
for g in G3_labels:
    pd.DataFrame(rows_M3[g]).to_excel(recommendations_path / f"recommendations_M3_group{g}_{timestamp}.xlsx", index=False)
for g in G4_labels:
    pd.DataFrame(rows_M4[g]).to_excel(recommendations_path / f"recommendations_M4_group{g}_{timestamp}.xlsx", index=False)
for g in G5_labels:
    pd.DataFrame(rows_M5[g]).to_excel(recommendations_path / f"recommendations_M5_group{g}_{timestamp}.xlsx", index=False)

print("STEP 16 done. Containers ready for STEP 18 evaluation:")
print(f"recs_41: {len(recs_41)} | recs_42: {len(recs_42)}")
print(f"M3 groups: {[len(recs_M3[g]) for g in G3_labels]}")
print(f"M4 groups: {[len(recs_M4[g]) for g in G4_labels]}")
print(f"M5 groups: {[len(recs_M5[g]) for g in G5_labels]}")
#%% STEP 18: EVALUATE (4.1/4.2/4.3) AND BUILD TABLES
print("========== STEP 18: EVALUATE (4.1/4.2/4.3) ==========")

# =============================================================================
# 1) Build ground truth from D_lst (single actions only)
# =============================================================================
D_triplets = erst.build_D_triplets_from_Dlst(D_lst)

print(f"D_triplets (single actions) size: {len(D_triplets)}")
print(f"Eval params: top_n_groundtruth={top_n_groundtruth} | k_eval={top_k_final_recommendations}")

# Quick sanity checks
if len(recs_41) == 0:
    print("WARNING: recs_41 is empty -> baseline metrics will be 0.")
if len(recs_42) == 0:
    print("WARNING: recs_42 is empty -> context metrics will be 0.")

# =============================================================================
# 2) 4.1 Baseline metrics
# =============================================================================
p41, r41, f41 = erst.evaluate_recommender_metrics_filtered_groundtruth(
    D_triplets,
    recs_41,
    top_n_groundtruth=top_n_groundtruth,
    k_eval=top_k_final_recommendations,
    groundtruth_filter_fn=None
)
avg_score_41 = float(np.mean([s for _, _, s in recs_41])) if len(recs_41) else 0.0

print(f"[4.1] Precision={p41} Recall={r41} F1={f41} AvgScore={avg_score_41:.3f}")

# =============================================================================
# 4.1 EXPORT LaTeX TABLE (single-row) for chapter 4.1
# =============================================================================
if hasattr(erst, "save_df_as_latex_table"):
    try:
        df_eval_41 = pd.DataFrame([{
            "Category": "Brez konteksta (baseline)",
            "Precision": p41,
            "Recall": r41,
            "F1": f41,
            "AverageScore": round(avg_score_41, 3),
        }])

        erst.save_df_as_latex_table(
            df=df_eval_41,
            out_dir=tabs_path,
            filename_stem=f"evaluation_metrics_4_1_{timestamp}",
            caption="Evalvacijske metrike priporočilnega sistema brez konteksta (4.1).",
            label="tab:evaluation_metrics_4_1",
            float_format="{:.3f}",
            index=False
        )
        print(f"Saved LaTeX table (4.1) to: {tabs_path}")
    except Exception as e:
        print(f"WARNING: LaTeX export (4.1) failed: {e}")

# =============================================================================
# 3) 4.2 Context metrics
# =============================================================================
p42, r42, f42 = erst.evaluate_recommender_metrics_filtered_groundtruth(
    D_triplets,
    recs_42,
    top_n_groundtruth=top_n_groundtruth,
    k_eval=top_k_final_recommendations,
    groundtruth_filter_fn=None
)
avg_score_42 = float(np.mean([s for _, _, s in recs_42])) if len(recs_42) else 0.0

print(f"[4.2] Precision={p42} Recall={r42} F1={f42} AvgScore={avg_score_42:.3f}")

# =============================================================================
# 4.2 EXPORT LaTeX TABLE (single-row) for chapter 4.2
# =============================================================================
if hasattr(erst, "save_df_as_latex_table"):
    try:
        df_eval_42 = pd.DataFrame([{
            "Category": "S kontekstom (C\\_T, C\\_P)",
            "Precision": p42,
            "Recall": r42,
            "F1": f42,
            "AverageScore": round(avg_score_42, 3),
        }])

        erst.save_df_as_latex_table(
            df=df_eval_42,
            out_dir=tabs_path,
            filename_stem=f"evaluation_metrics_4_2_{timestamp}",
            caption="Evalvacijske metrike priporočilnega sistema s kontekstom (4.2).",
            label="tab:evaluation_metrics_4_2",
            float_format="{:.3f}",
            index=False
        )
        print(f"Saved LaTeX table (4.2) to: {tabs_path}")
    except Exception as e:
        print(f"WARNING: LaTeX export (4.2) failed: {e}")

# =============================================================================
# 4) 4.3 Grouped metrics with filtered ground-truth
# =============================================================================
def make_gt_filter_for_group(actID_to_group, g):
    """
    Keep only ground-truth items that belong to group g.
    act_seq is (actID,) -> act_id = act_seq[0]
    """
    def _fn(act_seq):
        act_id = str(act_seq[0])
        if act_id not in actID_to_group:
            return False
        try:
            return int(actID_to_group[act_id]) == int(g)
        except Exception:
            return str(actID_to_group[act_id]) == str(g)
    return _fn

def eval_grouped_avg(recs_by_group, actID_to_group, labels, layer_name):
    """
    Returns:
    - df_per_group: per-group metrics
    - avg_dc: Option A = average across groups (unweighted mean)
    """
    rows = []
    for g in labels:
        p, r, f = erst.evaluate_recommender_metrics_filtered_groundtruth(
            D_triplets,
            recs_by_group[g],
            top_n_groundtruth=top_n_groundtruth,
            k_eval=top_k_final_recommendations,
            groundtruth_filter_fn=make_gt_filter_for_group(actID_to_group, g)
        )
        avg_s = float(np.mean([s for _, _, s in recs_by_group[g]])) if len(recs_by_group[g]) else 0.0

        rows.append({
            "Layer": layer_name,
            "Group": g,
            "Precision": p,
            "Recall": r,
            "F1": f,
            "AverageScore": round(avg_s, 3),
            "NumRecs": len(recs_by_group[g]),
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        avg_dc = {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "AverageScore": 0.0}
    else:
        avg_dc = {
            "Precision": round(float(df["Precision"].mean()), 3),
            "Recall": round(float(df["Recall"].mean()), 3),
            "F1": round(float(df["F1"].mean()), 3),
            "AverageScore": round(float(df["AverageScore"].mean()), 3),
        }

    return df, avg_dc

# M3 / M4 / M5 (Option A = avg across groups)
df_m3, m3_avg = eval_grouped_avg(recs_M3, actID_to_g3, G3_labels, "M3")
df_m4, m4_avg = eval_grouped_avg(recs_M4, actID_to_g4, G4_labels, "M4")
df_m5, m5_avg = eval_grouped_avg(recs_M5, actID_to_g5, G5_labels, "M5")

print(f"[M3 avg] {m3_avg}")
print(f"[M4 avg] {m4_avg}")
print(f"[M5 avg] {m5_avg}")

# =============================================================================
# 5) Save per-group evaluation (appendix / debugging)
# =============================================================================
out_m3 = recommendations_path / f"evaluation_metrics_M3_per_group_{timestamp}.xlsx"
out_m4 = recommendations_path / f"evaluation_metrics_M4_per_group_{timestamp}.xlsx"
out_m5 = recommendations_path / f"evaluation_metrics_M5_per_group_{timestamp}.xlsx"

df_m3.to_excel(out_m3, index=False)
df_m4.to_excel(out_m4, index=False)
df_m5.to_excel(out_m5, index=False)

print(f"Saved: {out_m3}")
print(f"Saved: {out_m4}")
print(f"Saved: {out_m5}")

# =============================================================================
# 6) Build Table 4.5 (5 rows) + export
# =============================================================================
table_45 = pd.DataFrame([
    {"Category": "Brez konteksta (baseline)", "Precision": p41, "Recall": r41, "F1": f41, "AverageScore": round(avg_score_41, 3)},
    {"Category": "S kontekstom (C_T, C_P)",  "Precision": p42, "Recall": r42, "F1": f42, "AverageScore": round(avg_score_42, 3)},
    {"Category": "M3 (avg čez 3 grupe)",     "Precision": m3_avg["Precision"], "Recall": m3_avg["Recall"], "F1": m3_avg["F1"], "AverageScore": m3_avg["AverageScore"]},
    {"Category": "M4 (avg čez 4 grupe)",     "Precision": m4_avg["Precision"], "Recall": m4_avg["Recall"], "F1": m4_avg["F1"], "AverageScore": m4_avg["AverageScore"]},
    {"Category": "M5 (avg čez 5 grup)",      "Precision": m5_avg["Precision"], "Recall": m5_avg["Recall"], "F1": m5_avg["F1"], "AverageScore": m5_avg["AverageScore"]},
])

out_table = recommendations_path / f"table_4_5_all_layers_{timestamp}.xlsx"
table_45.to_excel(out_table, index=False)

print("\n========== TABLE 4.5 (ALL LAYERS) ==========")
print(table_45)
print(f"Saved: {out_table}")

# =============================================================================
# 7) EXPORT TABLE 4.5 AS LaTeX (in correct output directory)
# =============================================================================
if hasattr(erst, "save_df_as_latex_table"):
    try:
        erst.save_df_as_latex_table(
            df=table_45,
            out_dir=tabs_path,  # Save to LaTeX tables directory
            filename_stem=f"table_4_5_all_layers_{timestamp}",
            caption="Primerjava uspešnosti priporočilnega sistema pri različnih plasteh konteksta.",
            label="tab:table_4_5",
            float_format="{:.3f}",
            index=False
        )
        print(f"Saved LaTeX table to: {tabs_path}")
    except Exception as e:
        print(f"WARNING: LaTeX export failed: {e}")

# =============================================================================
# 8) PLOT: Precision / Recall / F1 comparison (4.1 vs 4.2)
# =============================================================================

labels = ["Precision", "Recall", "F1"]

no_ctx = [
    float(p41),
    float(r41),
    float(f41),
]

ctx = [
    float(p42),
    float(r42),
    float(f42),
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(7, 4))
plt.bar(x - width/2, no_ctx, width, label="Brez konteksta")
plt.bar(x + width/2, ctx, width, label="S kontekstom")

plt.xticks(x, labels)
plt.ylabel("Vrednost metrike")
plt.title("Primerjava ranking metrik: brez konteksta vs s kontekstom")
plt.legend()
plt.tight_layout()

fig_path = figs_path / f"prf_comparison_{timestamp}.png"
plt.savefig(fig_path)
plt.show()

print(f"Saved PRF comparison plot to: {fig_path}")

# %%
