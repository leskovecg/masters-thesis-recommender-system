"""
EXPLAINABLE ELDERLY RECOMMENDER SYSTEM PIPELINE
===================================================

This script implements a modular and explainable recommender system for suggesting daily activities to elderly users.
It integrates multiple data sources (user responses, health dimensions, contextual metadata) and applies collaborative filtering 
(SVD, KNN, NMF, BaselineOnly) to build personalized and context-aware recommendations.

The main components of the pipeline are:

STEP 0: Importing required libraries
STEP 1: Configuration & parameter settings
STEP 2: Loading user and activity data from Excel
STEP 3: Building dictionaries from context definitions
STEP 4: Selecting actions based on selected aspects
STEP 5: Generating simplified context strings
STEP 6: Generating sequences of actions
STEP 7: Trimming users and actions (for testing)
STEP 8: Computing action relevance scores
STEP 9: Filtering the compatibility matrix
STEP 10: Building the user-action response matrix
STEP 11: Building D_lst data matrix for training
STEP 12: Comparing multiple algorithms using Surprise
STEP 13: Hyperparameter tuning (GridSearch for SVD)
STEP 14: Cross-validation for tuned SVD
STEP 15: Final model training and matrix factorization
STEP 16: Generating explainable context-aware recommendations
STEP 17: Exporting recommendations
STEP 18: Evaluating recommendations with Precision / Recall / F1
STEP 19: Printing samples and debugging outputs
"""

#%% STEP 0: IMPORT LIBRARIES
##==================================================================================
print("========== STEP 0: IMPORTING LIBRARIES ==========")

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
import time
from tqdm import tqdm
from collections import Counter

#%% STEP 1: CONFIGURATION & SETTINGS
##==================================================================================
print("========== STEP 1: GLOBAL SETTINGS ==========")

test_mode = False                # Enable test mode for debugging

n_splits = 5                    # Number of splits for cross-validation
M = 100                           # Number of latent features for matrix factorization (dummy value here)
top_n_groundtruth = 5           # How many top ground-truth items to use for evaluation
n_recommendations = 20          # Number of top candidates returned (before filtering)
top_k_final_recommendations = 5 # Number of final context-aware recommendations

BASE_DIR = Path(__file__).resolve().parents[1]
data_path = BASE_DIR / 'data'
tabs_path = BASE_DIR / 'latex' / 'tabs'
figs_path = BASE_DIR / 'latex' / 'figs'

# Generate a timestamp (YYYYMMDD_HHMMSS) to version output files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#%% STEP 2: LOAD DATA FROM EXCEL
##==================================================================================
print("========== STEP 2: LOADING DATA FROM EXCEL FILES ==========")

# Load activity data, replace -1 with NaN, and rename columns with 'Ac_' prefix
activities_df = pd.read_excel(data_path / 'Activities.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
activities_df.rename(columns={nm:'Ac_' + nm for nm in activities_df.columns}, inplace=True)

# Load mental health data, replace -1 with NaN, and rename columns with 'Mh_' prefix
mentalHealth_df = pd.read_excel(data_path / 'MentalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
mentalHealth_df.rename(columns={nm:'Mh_' + nm for nm in mentalHealth_df.columns}, inplace=True)

# Load physical health data, replace -1 with NaN, and rename columns with 'Ph_' prefix
physicalHealth_df = pd.read_excel(data_path / 'PhysicalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
physicalHealth_df.rename(columns={nm:'Ph_' + nm for nm in physicalHealth_df.columns}, inplace=True)

# Load social health data, replace -1 with NaN, and rename columns with 'Sh_' prefix
socialHealth_df = pd.read_excel(data_path / 'SocialHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
socialHealth_df.rename(columns={nm:'Sh_' + nm for nm in socialHealth_df.columns}, inplace=True)

# Load context definitions for activities (time and place conditions)
activityContextGen_df = pd.read_excel(data_path / 'ActivityContextGen_v09.xlsx', sheet_name='ActionLst').replace(-1, np.nan)

# Load user-specific scores and weights (how suitable each activity is for each user)
scores_and_wgt_df = pd.read_excel(data_path / 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')

# Optional: other versions of the score data (commented out for now)
#scores_and_wgt_2_df = pd.read_excel(data_path + 'ml_data_with_scores_and_wgt_3_values.xlsx', sheet_name='scores_and_wgt_3_v', header=1)
#wgt_results_annotations_df = pd.read_excel(data_path + 'wgt_results_annotations_3009_4users.xlsx', sheet_name='wgt_results_annotations_3009_4u', header=1).drop(columns=['Unnamed: 0', 'Column1'])

# Merge all user responses (activities + mental + physical + social) into a single DataFrame
all_answers_df = activities_df.join(mentalHealth_df).join(physicalHealth_df, rsuffix='_r').join(socialHealth_df, rsuffix='_r')

# Load handcrafted compatibility matrix (with continuous values between 0 and 1)
action_compat_df = pd.read_excel(data_path / 'ActivityContextGen_v10.xlsx', sheet_name='ActionCompatibilities', index_col=0)

#%% STEP 3: BUILD DICTIONARIES & STRUCTURES
##==================================================================================
print("========== STEP 3: BUILDING DICTIONARIES & STRUCTURES ==========")

# Extract user IDs from the scores DataFrame
uIDs = list(np.sort(scores_and_wgt_df.index))

# Define lists of question IDs for different health/activity aspects
# These group questionnaire columns relevant to each domain
# These lists group specific columns (questions) related to activities, physical health, mental health, and social health. Each list contains column names that are relevant to the respective group.
activity_qs = ['Ac_AB4_1', 'Ac_AB4_2', 'Ac_AB4_3', 'Ac_AB4_4', 'Ac_AB4_5', 'Ac_AB4_8'] 
#'activity' = [AB4_1, AB4_2, AB4_3, AB4_4, AB4_5, AB4_8]
phy_health_qs = ['Ph_AB1_7', 'Ph_AB1_11', 'Ph_AB3', 'Ph_AB6_1', 'Ph_AB6_5', 'Ph_AB7_1', 'Ph_AB7_5', 'Ph_AB4_2', 'Ph_AB4_3', 'Ph_AB4_4']
#'pyhisicalHealth ' = [AB1_11, AB3, AB6_1, AB6_5, AB7_1, AB7_5, AB4_2, AB4_3, AB4_4]
ment_helath_qs = ['Mh_A75_2', 'Mh_A75_3', 'Mh_A75_4', 'Mh_A75_5', 'Mh_AB1_14', 'Mh_A82_r1', 'Mh_A82_r3']
# 'mental_health' = [A75_2, A75_3, A75_4, A75_5, AB1_14, A82_r1, A82_r3, A83_r]
soc_helath_qs = ['Sh_A83_r', 'Sh_sh_AB98_da_ne']

# This dictionary groups the question lists into a single structure, allowing easy access to each group by its category name.
group_qLst = {'activity':activity_qs,
              'phy_health':phy_health_qs,
              'ment_helath':ment_helath_qs,
              'soc_helath':soc_helath_qs}

# This dictionary maps different health and activity groups to specific factors (like F1, F2, etc.). These factors likely represent different components or categories relevant for the analysis.
group_factor_dc = {'Activities': 'F2', 
                   'PhysicalHealth-organskiSistemi': 'F1',
                   'PhysicalHealth-nacinZivljenja': 'F1',
                   'MentalHealth-osnovno': 'F1',
                   'MentalHealth-visje': 'F3'}

# Create dictionaries to both ways -  This helps in easily finding corresponding actions based on their IDs and vice versa.
actID_singleAct_dc = dict(zip(activityContextGen_df['actID'], activityContextGen_df['Single_action']))
singleAct_actID_dc = dict(zip(activityContextGen_df['Single_action'], activityContextGen_df['actID']))

# Initialize Dictionaries and Variables: The following dictionaries and variables are used to store mappings between question IDs, actions, and their properties:
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
    
    # New qID
    # Skip certain question IDs
    if row['qID'] not in ['A82_r1', 'A82_r3', 'A83_r']: # Not covered Qs

        if pd.notnull(row['qID']):
            if row['qID'] not in curr_qIDs:
                curr_qID = row['qID']
                curr_qIDs.append(curr_qID)
                qID_qtxt_dc[curr_qID] = row['qText']
                qID_Group_dc[curr_qID] = row['Group']
                qID_singleAct_dc[curr_qID] = []
                qID_actID_dc[curr_qID] = []

        last_qID = curr_qIDs[-1]

        # Add vals
        # Link actions and IDs to the question
        single_act = row['Single_action']
        actID = row['actID']
        qID_singleAct_dc[curr_qID].append(single_act)
        #singleAct_qID_dc[curr_qID+'_'+str(single_act)] = curr_qID
        singleAct_qID_dc.update({curr_qID+'_'+str(single_act):curr_qID})
        qID_actID_dc[curr_qID].append(actID)
        actID_qID_dc.update({str(actID):curr_qID})

        # Store action properties (e.g., usefulness, difficulty, etc.)
        actID_props_dc[actID] = {}
        actID_props_dc[actID]['qID'] = last_qID
        actID_props_dc[actID]['action_prop_1'] = row['action_prop_1']
        actID_props_dc[actID]['action_prop_2'] = row['action_prop_2']
        actID_props_dc[actID]['action_prop_3'] = row['action_prop_3']

        # Store action context (up to 3 time and 3 place descriptors)
        actID_context_dc[actID] = {}
        actID_context_dc[actID]['qID'] = last_qID
        actID_context_dc[actID]['act_C_T1'] = row['act_C_T1']
        actID_context_dc[actID]['act_C_T2'] = row['act_C_T2']
        actID_context_dc[actID]['act_C_T3'] = row['act_C_T3']
        actID_context_dc[actID]['act_C_P1'] = row['act_C_P1']
        actID_context_dc[actID]['act_C_P2'] = row['act_C_P2']
        actID_context_dc[actID]['act_C_P3'] = row['act_C_P3']

# uIDs to socres
# Convert scores for each user into dictionaries for each aspect
uID_activity_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['a_F2']))
uID_menHealOsn_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_osnovno_F1']))
uID_menHealVisje2_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F2']))
uID_menHealVisje4_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F4']))
uID_phyHealNacin_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_nacinZivljenja_F1']))
uID_phyHealOrganski_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_organskiSistemi_F1']))

# Aggregate all score dictionaries into one by category

# Select which aspect groups to include in the recommendation process
# Available options include: 'activity', 'phy_health', 'ment_helath', 'soc_helath'
# Here, only the 'activity' aspect is used for simplicity
aspect_groups_lst = ['activity'] #, 'phy_health'] #, 'ment_helath', 'soc_helath']

#score_groups_lst = ['activity', 'menHealOsn', 'menHealVisje2', 'menHealVisje4', 'phyHealNacin', 'phyHealOrganski']
uID_scores_dc = {}
uID_scores_dc['activity'] = uID_activity_scores_dc
uID_scores_dc['menHealOsn'] = uID_menHealOsn_scores_dc
uID_scores_dc['menHealVisje2'] = uID_menHealVisje2_scores_dc
uID_scores_dc['menHealVisje4'] = uID_menHealVisje4_scores_dc
uID_scores_dc['phyHealNacin'] = uID_phyHealNacin_scores_dc
uID_scores_dc['phyHealOrganski'] = uID_phyHealOrganski_scores_dc

# Display the activity score dictionary for inspection (optional line)
# uID_scores_dc['activity']

#%% STEP 4: SELECT ACTIONS FOR GIVEN ASPECTS
##==================================================================================
print("========== STEP 4: SELECTING ACTIONS FOR GIVEN ASPECTS ==========")

"""
We filter actions and actIDs based on the selected aspect group(s),
e.g. only actions related to the 'activity' aspect.
"""

#single_act_lst = [g for g in singleAct_qID_dc]

# Generate a list of single actions relevant to the selected aspect group(s)
# For example, if 'activity' is selected, it includes only actions tied to 'Ac_AB4_x' questions
# singleAct_qID_dc maps "qID + action" => qID
single_act_lst = []
for group in aspect_groups_lst:
    for act in singleAct_qID_dc:
        for qID in group_qLst[group]:
            if qID in act:
                single_act_lst.append(act)

#single_actID_lst = [g for g in actID_qID_dc]
# Generate a list of actIDs relevant to the selected aspect group(s)
# actID_qID_dc maps actID => qID; we filter actIDs whose qID is in the selected group
single_actID_lst = []
for group in aspect_groups_lst:
    for actID in actID_qID_dc:
        for qID in group_qLst[group]:
            if qID in actID:
                single_actID_lst.append(actID)

#%% STEP 5: PROCESS CONTEXT STRINGS
##==================================================================================
print("========== STEP 5: PROCESS CONTEXT STRINGS ==========")

"""
Generates a simplified 'kontekst' string from C_T1–C_T3 columns and trims the last invalid rows.
"""

# Generate a new column 'kontekst' by combining valid time context values (C_T1, C_T2, C_T3)
# The `get_context` function handles this by filtering and formatting time descriptions
activityContextGen_df['kontekst'] = activityContextGen_df.apply(
    lambda row: erst.get_context(row['act_C_T1'], row['act_C_T2'], row['act_C_T3']), axis=1
    )
# print(df[['Single_action', 'C_T1', 'C_T2', 'C_T3', 'kontekst']])

# Extract the context strings into a list for further use
context_lst = activityContextGen_df['kontekst'].tolist()

# Remove the last 6 elements (likely placeholders or incomplete entries)
context_lst = context_lst[:-6]

#%% STEP 6: GENERATE SEQUENCES
##==================================================================================
print("========== STEP 6: GENERATING SEQUENCES OF ACTIONS ==========")

"""
Creates all valid combinations of actions up to max length.
"""

# Define the maximum allowed length of action sequences
act_max_len = 3

# Generate all possible sequences of actions (represented by actIDs)
seq_actID_lst = erst.get_list_of_actions(single_actID_lst, act_max_len)

#%% STEP 7: TRIM USERS/ACTIONS FOR TESTING (reduced number of users and activities)
##==================================================================================
print("========== STEP 7: TRIM USERS/ACTIONS FOR TESTING ==========")

"""
For faster evaluation, we limit how many users/actions we consider.
"""

# Settings for limiting the number of users and actions considered in the matrix.
# uIDs_n, acts_n = 100, 80    # Number of users and actions to consider
uIDs_n, acts_n = len(uIDs), len(seq_actID_lst)       # Limit number of users and actions for faster testing

# Select subset of users and action sequences
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n] 
#uIDsIn, seq_actID_lstIn = uIDs, seq_actID_lst

# Extract only single-action sequences for relevance computation.
actID_lstIn = [act[0] for act in seq_actID_lstIn if len(act)==1]

#%% STEP 8: COMPUTE ACTION RELEVANCE SCORES
##==================================================================================
print("========== STEP 8: COMPUTING ACTION RELEVANCE SCORES ==========")

"""
Calculates how relevant each action is for each user based on scores and answers.
"""

# Method for relevance computation and threshold for inclusion.
meth_code = 'score'
r_T = 0.3 

# Precomputed data frames
# Scores for each action and user based on their responses and associated scores.
actID_score_df = erst.get_actID_score_df(
    uIDsIn, actID_lstIn, actID_qID_dc, 
    uID_scores_dc, all_answers_df, 
    aspect_groups_lst, meth_code
    )

# TO SPODAJ JE AVTOMATSKO DELALO....
# # Compatibility data between pairs of actions, which likely evaluates how well different actions fit together.
# actID_compat_df = erst.get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)

print("action_compat_df.head():")
print(action_compat_df.head())
print("action_compat_df.index:")
print(action_compat_df.index)
print(len(action_compat_df.index))
print("actID_lstIn:")
print(actID_lstIn)
print(len(actID_lstIn))

#%% STEP 9: LOAD AND FILTER COMPATIBILITY MATRIX
##==================================================================================
print("========== STEP 9: LOAD AND FILTER COMPATIBILITY MATRIX ==========")

"""
We take the full handcrafted matrix and filter to keep only actions we're working with.
"""

# Filter handcrafted matrix to include only relevant actions
actID_compat_df = action_compat_df.loc[actID_lstIn, actID_lstIn]
print("\nFiltered action compatibility matrix (actID_compat_df):")
print(actID_compat_df.head())

#%% STEP 10: BUILD USER-ACTION RESPONSE MATRIX
##==================================================================================
print("========== STEP 10: BUILDING USER-ACTION RESPONSE MATRIX ==========")

"""
This builds a matrix where each cell is user's answer to the question linked to an action.
"""

# User responses to the questions linked to each action, used to compute the relevance of the action.
uID_qID_answers_df = erst.get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst)

# Build a DataFrame with users as rows and actions as columns.
# Each cell contains the user's response to the question associated with that action.
uID_actID_answers_df = pd.DataFrame(index=uID_qID_answers_df.index)
for actID in actID_lstIn:
    if actID_qID_dc[actID] in uID_qID_answers_df:
        uID_actID_answers_df[actID] = uID_qID_answers_df[actID_qID_dc[actID]]
    else:
        print ('Error:' + actID)

# Visualize Action-Relevance Heatmap (optional)

# Visualize the user responses to each question (alternative: uncomment for question-level heatmap)
# sns.heatmap(uID_qID_answers_df)

# Visualize the user responses to each action (rows = users, columns = actions)
# Each cell shows how relevant a given action is to a specific user based on their original response to the associated question
sns.heatmap(uID_actID_answers_df)
plt.title("User vs Action Relevance (based on questionnaire answers)")
plt.xlabel("Actions")
plt.ylabel("Users")
plt.tight_layout()
plt.savefig(figs_path / f"user_action_relevance_heatmap_{timestamp}.png")
plt.show()

#%% STEP 11: BUILD D_lst MATRIX
##==================================================================================
print("========== STEP 11: BUILDING D_lst MATRIX ==========")

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
# SAVE .pkl:
with open(data_path / d_lst_filename, "wb") as f:
    pickle.dump(D_lst, f)
print(f"Saved full D_lst to: {data_path / d_lst_filename}")

# # Mentor's new and laready built D_lst matrix: 
# with open(data_path/ 'D_contx_a3_sparse_mat.pkl', "rb") as fp:
#     D_lst = pickle.load(fp)

# D_lst = D_lst[:10000]  # Take only the first 10000 entries — for faster testing


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
print("\n Najnižje ocene:")
for x in sorted_by_rating[:3]:
    print(x)

print("\n Najvišje ocene:")
for x in sorted_by_rating[-3:]:
    print(x)

# Kontekstna analiza (C_T1 in C_P1)
context_counts = Counter(
    (x[2][0].get("C_T1", "None"), x[2][0].get("C_P1", "None"))
    for x in D_lst if isinstance(x[2], tuple) and isinstance(x[2][0], dict)
)

print("\n Top 10 kontekstov (C_T1, C_P1):")
for ctx, count in context_counts.most_common(10):
    print(f"{ctx}: {count}x")

# Vizualizacija
plt.hist(ratings, bins=20)
plt.title("Distribucija ocen v D_lst")
plt.xlabel("Ocena")
plt.ylabel("Število vnosov")
plt.grid(True)
plt.savefig(figs_path / f"ratings_distribution_hist_{timestamp}.png")
plt.show()

#%% STEP 12: ALGORITHM COMPARISON (SVD, KNNBasic, BaselineOnly, NMF)
##==================================================================================
print("========== STEP 12: ALGORITHM COMPARISON ==========")

"""
This step loads a sample of D_lst and prepares the rating data for Surprise.
Then, several collaborative filtering algorithms are evaluated using KFold cross-validation.
"""

# Load D_lst and prepare DataFrame 
# ========================================================== 

print("\n========== Loading D_lst... ==========")
with open(data_path / "D_lst_full.pkl", "rb") as f:
    D_lst = pickle.load(f)
# D_lst = D_lst[:1000]

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

print("\n Performing cross-validation for each algorithm...")
for name, algo_class in tqdm(algorithms.items(), desc="Evaluating Algorithms"):
    print(f"\n=== Evaluating {name} ===")
    cv_df = erst.perform_cross_validation(
        data=data,
        model_class=algo_class,
        algorithm_name=name,
        n_splits=n_splits,
        random_state=42
    )

    mean_metrics = cv_df[cv_df['Fold'] == 'Mean'].iloc[0]

    metrics_summary.append({
        'algorithm': name,
        'rmse': round(mean_metrics['RMSE'], 4),
        'mae': round(mean_metrics['MAE'], 4),
        'mse': round(mean_metrics['MSE'], 4),
        'fcp': round(mean_metrics['FCP'], 4)
    })

# Save results and plot
# ==========================================================

# Convert to DataFrame and export
summary_df = pd.DataFrame(metrics_summary)
print("\n Cross-validation results summary:")
print(summary_df)

summary_df.to_excel(tabs_path / f"algorithm_comparison_{timestamp}.xlsx", index=False)

plt.figure(figsize=(8,5))
plt.bar(summary_df["algorithm"], summary_df["rmse"], color="lightblue")
plt.ylabel("Average RMSE")
plt.title("Povprečni RMSE za algoritme")
plt.tight_layout()
plt.savefig(figs_path / f"algorithm_rmse_comparison_{timestamp}.png")
plt.show()

#%% STEP 13: GRID SEARCH & SVD TUNING
##==================================================================================
print("========== STEP 13: GRID SEARCH & SVD TUNING ==========")

"""
SVD model is tuned using GridSearch to find optimal hyperparameters.
"""

print("\n========== GRID SEARCH ==========")
# # Define hyperparameter search space for the SVD algorithm (from Surprise library)

param_grid = {
    'n_factors': [10] if test_mode else [50, 100, 150],
    'n_epochs': [5] if test_mode else [20, 30],
    'lr_all': [0.002] if test_mode else [0.002, 0.005],
    'reg_all': [0.02] if test_mode else [0.02, 0.1]
}

profile = cProfile.Profile() 
profile.enable() 

# Run grid search to find the best combination of hyperparameters
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=n_splits)
gs.fit(data)

best_params = gs.best_params['rmse']
print(f"Best params: {best_params}")
print("Best RMSE score:", gs.best_score['rmse'])

profile.disable()
# profile.print_stats(sort='cumtime')  

#%% STEP 14: CROSS-VALIDATION FOR TUNED SVD
##==================================================================================
print("========== STEP 14: CROSS-VALIDATION (TUNED SVD) ==========")

"""
This step validates the performance of the tuned SVD model using n_splits-fold cross-validation.
"""

# === Run cross-validation with tuned parameters ===
print(f"\n Evaluating tuned SVD model with {n_splits}-fold CV...")
profile = cProfile.Profile()
profile.enable()
cv_results_df = erst.perform_cross_validation(
    data=data, 
    model_class=lambda: SVD(**best_params),  # factory for new model
    n_splits=n_splits, 
    random_state=42
)
profile.disable()
# profile.print_stats(sort='cumtime')

cv_results_df.to_excel(tabs_path / f"cv_results_tuned_SVD_{timestamp}.xlsx", index=False)

# === Extract and display average metrics ===
print("\n Cross-validation results for tuned SVD:")
print(cv_results_df)

# Extract mean values from the 'Mean' row
mean_metrics = cv_results_df[cv_results_df['Fold'] == 'Mean'].iloc[0]

# Prepare a summary list for display
results_list = [
    {
        'Algorithm': 'SVD',
        'Average RMSE': mean_metrics['RMSE'],
        'Average MAE': mean_metrics['MAE'],
        'Average MSE': mean_metrics['MSE'],
        'Average FCP': mean_metrics['FCP'],
        'Average Training Time': mean_metrics['Fit time']
    }
]

# print("\nAverage metrics across folds:")
# print(results_list)

#%% STEP 15: FINAL TRAINING & EVALUATION (TRAIN FINAL MODEL ON FULL DATASET OR SPLIT DATASET)
##==================================================================================
print("========== STEP 15: FINAL TRAINING & EVALUATION ==========")

"""
Train the final SVD model on all data and evaluate its performance.
"""

# # Train the model on the full training set
# trainset = data.build_full_trainset()

# Train the model on the training set
trainset, testset = train_test_split(data, test_size=0.2)

print("\n========== TRAIN(TEST)SET INFO ==========")
print(f"Number of users in trainset: {trainset.n_users}")
print(f"Number of items in trainset: {trainset.n_items}")
# print(f"Trainset length: {len(trainset)}")
print(f"Trainset length: {trainset.n_ratings}")

print(f"Testset length: {len(testset)}")
print(f"Total unique actions in D_lst: {len(set(x[1] for x in D_lst))}")
print(dir(trainset))


# Initialize the model with the best parameters
model = SVD(**best_params)

profile = cProfile.Profile()
profile.enable()

# Train the algorithm on the trainset
model.fit(trainset)

# predict ratings for the testset
predictions = model.test(testset)

print("\n========== MATRIX FACTORIZATION FACTORS ==========")
print(f"Shape of user factors: {model.pu.shape}")
print(f"Shape of item factors: {model.qi.shape}")

profile.disable()
# profile.print_stats(sort='cumtime')

# Compute basic metrics on the test sets
print("\n========== TEST SET EVALUATION ==========")
print(f"RMSE: {accuracy.rmse(predictions)}")
print(f"MAE:  {accuracy.mae(predictions)}")
print(f"MSE:  {accuracy.mse(predictions)}")
print(f"FCP:  {accuracy.fcp(predictions)}")

#%% STEP 16: GENERATE CONTEXT-AWARE RECOMMENDATIONS
##==================================================================================
print("========== STEP 16: GENERATE CONTEXT-AWARE RECOMMENDATIONS ==========")

"""
For each user:
- recommend top N actions based on model
- filter based on context feasibility
- save top 5 with explanations (answers, scores, MF vectors)
"""

# EXTRACT USER AND ITEM FACTORS (P, Q)
# ==========================================================

# Build user and item factor dicts
user_factors = {trainset.to_raw_uid(uid): model.pu[uid] for uid in range(trainset.n_users)}
item_factors = {trainset.to_raw_iid(iid): model.qi[iid] for iid in range(trainset.n_items)}
print("\n========== FACTORS INFO ==========")
print(f"User factors: {len(user_factors)} extracted")
print(f"Item factors: {len(item_factors)} extracted")


# GENERATE EXPLAINABLE RECOMMENDATIONS FOR EACH USER
# ===========================================================

"""
This section prepares a detailed export of recommended actions along with explanations 
for why those actions are relevant to each user.

What is included in the export:
 - uID: user ID
 - One recommended action per row (multiple recommendations per user, so uID may repeat)
 - Explanations include:
   - Context: a relevant situation for the action 
   - User’s answer for the question linked to the action
   - User’s overall score for this aspect (e.g., physical activity)
   - Explainability components from matrix factorization: latent feature vectors P and Q

The goal is to construct interpretable recommendations such as:
  - context: "It's nice weather and the right time to go outside"
  - qaID: "You’ve previously shown a strong preference for this activity"
  - scores: "You're physically capable and active, so this fits you well"
  - P and Q: "Your profile aligns with others who also enjoy this action (based on P and Q vectors)"
"""

dc_lst_41 = []   # for section 4.1
dc_lst_42 = []   # for section 4.2
all_recommendations = []           # top-20 brez konteksta
all_final_recommendations = []     # top-5 filtriranih s kontekstom

# Define an example context for generating recommendations 
context = {'C_T': 'ob kosilu', 'C_P': 'kjerkoli'}

for uID in tqdm(uIDsIn, desc="Generating recommendations"):

    print(f"\n========== Generating recommendations for user {uID} ==========")

    # get qaIDs
    c_anws_a = dict(uID_qID_answers_df.loc[uID,:])
    print(f"User context answers keys: {list(c_anws_a.keys())}")

    # Get corresponding answer texts
    c_anws_txt = {qID:qID_qtxt_dc[qID] for qID in c_anws_a}

    # Get user score for the current aspect (e.g., 'activity')
    c_score = uID_scores_dc['activity'][uID]
    print(f"User score on activity: {c_score}")

    # Dummy placeholders for Matrix Factorization vectors (can be replaced with real ones)
    # get P and Q
    # c_MF_P = [1,2,3]
    # c_MF_Q = [3,2,1]


    # Get real P and Q
    try:
        c_MF_P = user_factors[uID]
    except KeyError:
        c_MF_P = [0.0] * M  # fallback


    # GET RECOMMENDATIONS FOR ONE USER (WITHOUT CONTEXT, WITHOUT CONTEXT - USING SURPRISE LIBRARY AND WITH CONTEXT)
    # Get top `m` recommendations for the current user - # GET RECOMMENDATIONS - WITHOUT CONTEXT
    # best_act_trp_lst = erst.get_recommendations(uID=uID, D_lst=D_lst, n_recommendations=n_recommendations)
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, n_recommendations=n_recommendations)
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, context=context, actID_context_dc=actID_context_dc, n_recommendations=m)


    # # GET RECOMMENDATIONS - WITHOUT CONTEXT - USING SURPRISE LIBRARY
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, n_recommendations=n_recommendations)

    # # GET RECOMMENDATIONS - WITH CONTEXT
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, 
    #                 context=context, actID_context_dc=actID_context_dc, n_recommendations=n_recommendations)


    # first get top 20 candidates without filtering
    top20_recs = erst.get_recommendations(
        uID=uID,
        trainset=trainset,
        model=model,
        n_recommendations=n_recommendations
    )

    # then filter by context and take top 5
    filtered_recs = [
        rec for rec in top20_recs
        if erst.is_action_context_feasibleQ(rec[1], context, actID_context_dc)
    ]
    final_recs = filtered_recs[:top_k_final_recommendations]

    print(f"\n--- Kandidatke za uporabnika {uID} (top {n_recommendations}) brez konteksta ---")
    for i, (uid, act_id, score) in enumerate(top20_recs[:top_k_final_recommendations]):
        print(f"{i+1}. {act_id} (score: {score:.3f})")

    print(f"--- Končna priporočila za uporabnika {uID} (m = {top_k_final_recommendations}) po kontekstu ---")
    for i, (uid, act_id, score) in enumerate(final_recs):
        print(f"{i+1}. {act_id} (score: {score:.3f})")


    # all_recommendations.extend(best_act_trp_lst)  # <- Dodamo v glavni seznam

    # Pred tem:
    # all_recommendations.extend(best_act_trp_lst)

    # Instead of this:
    # all_recommendations.extend([(uID, act_seq, score) for act_seq, score in best_act_trp_lst])

    # all_recommendations.extend([
    #     (uid, act_seq, score) for uid, act_seq, context, score in best_act_trp_lst
    # ])

    # all_recommendations.extend(top20_recs)
    # all_final_recommendations.extend(final_recs)

    # Convert action string to tuple to match D_lst format
    all_recommendations.extend([(uid, (act_id,), score) for uid, act_id, score in top20_recs])
    all_final_recommendations.extend([(uid, (act_id,), score) for uid, act_id, score in final_recs])

    # print(f"\n========== TOP RECOMMENDATIONS FOR USER {uID} ==========")
    # for rec in best_act_trp_lst[:3]:  # limit to avoid flooding the output
    #     print(f"Action: {rec[1]}, Score: {rec[-1]:.3f}")

    for uid, act_id, score in top20_recs:
        # try:
        #     c_MF_Q = item_factors[act_id]
        # except KeyError:
        #     c_MF_Q = [0.0] * M
        
        c_MF_Q = next(
            (vec for key, vec in item_factors.items() if act_id in key),
            [0.0] * M  # fallback if not found
        )

        cntx_data = actID_context_dc.get(act_id, {})
        c_cntx = erst.get_one_random_context(cntx_data)

        dc_lst_41.append({
            'uID': uid,
            'ActID': act_id,
            'Score': score,
            'Context_T': c_cntx['act_C_T'],
            'Context_P': c_cntx['act_C_P'],
            'Anws_a': c_anws_a,
            'Anws_txt': c_anws_txt,
            'UserScore': c_score,
            'MF_P': c_MF_P,
            'MF_Q': c_MF_Q
        })
    
    # for section 4.2 (context-based)
    for uid, act_id, score in final_recs:
        # try:
        #     c_MF_Q = item_factors[act_id]
        # except KeyError:
        #     c_MF_Q = [0.0] * M

        c_MF_Q = next(
            (vec for key, vec in item_factors.items() if act_id in key),
            [0.0] * M  # fallback if not found
        )
        cntx_data = actID_context_dc.get(act_id, {})
        c_cntx = erst.get_one_random_context(cntx_data)
        dc_lst_42.append({
            'uID': uid,
            'ActID': act_id,
            'Score': score,
            'Context_T': c_cntx['act_C_T'],
            'Context_P': c_cntx['act_C_P'],
            'Anws_a': c_anws_a,
            'Anws_txt': c_anws_txt,
            'UserScore': c_score,
            'MF_P': c_MF_P,
            'MF_Q': c_MF_Q
        }) 

#%% STEP 17: EXPORT RECOMMENDATIONS
##==================================================================================
print("========== STEP 17: EXPORT RECOMMENDATIONS ==========")

"""
Save two recommendation tables (with and without context filtering) to Excel
"""

rec_X_df_41 = pd.DataFrame(dc_lst_41)
rec_X_df_41.to_excel(tabs_path / f"recommendations_4_1_{timestamp}.xlsx", index=False)

rec_X_df_42 = pd.DataFrame(dc_lst_42)
rec_X_df_42.to_excel(tabs_path / f"recommendations_4_2_{timestamp}.xlsx", index=False)

print(f"Exported {len(rec_X_df_41)} rows for 4.1 recommendations to recommendations_4_1_{timestamp}.xlsx")
print(f"Exported {len(rec_X_df_42)} rows for 4.2 recommendations to recommendations_4_2_{timestamp}.xlsx")

#%% STEP 18: EVALUATE PRECISION / RECALL / F1
##==================================================================================
print("========== STEP 18: EVALUATE PRECISION / RECALL / F1 ==========")

"""
Compute standard IR metrics (Precision, Recall, F1) by comparing model recs with ground truth D_lst
"""

# Convert D_lst from quadruples to triplets as expected by the function
D_triplets = [(uid, iid, rating) for uid, iid, context, rating in D_lst]

print("Sample from D_lst (ground truth):", D_triplets[0][1], type(D_triplets[0][1]))
print("Sample from all_recommendations:", all_recommendations[0][1], type(all_recommendations[0][1]))

print("\n=== Evaluacija priporočil ===")
print(f"Stevilo ground truth akcij (top_n_groundtruth): {top_n_groundtruth}")
print(f"Stevilo kandidatk brez konteksta (n_recommendations): {n_recommendations}")
print(f"Stevilo končnih priporočil (m): {top_k_final_recommendations}")
print("\n=== Ground Truth ===")
for uid in list(set([r[0] for r in D_triplets]))[:3]:  # for the first 3 users
    top_gt = sorted([r for r in D_triplets if r[0] == uid], key=lambda x: x[2], reverse=True)[:top_n_groundtruth]
    print(f"Uporabnik {uid} – top {top_n_groundtruth} ocenjenih aktivnosti:")
    for rec in top_gt:
        print(f"  {rec[1]} (ocena: {rec[2]:.2f})")


# 4.1: brez konteksta (standardna matrika)
##############################################################
p41, r41, f41 = erst.evaluate_recommender_metrics(
    D_triplets,
    all_recommendations,   # these are your non-contextual recommendations
    top_n_groundtruth=top_n_groundtruth,
    k_eval=top_k_final_recommendations
)

if all_recommendations:
    avg_score_41  = sum(rec[-1] for rec in all_recommendations) / len(all_recommendations)
    print(f"Average score: {avg_score_41 :.3f}")
    print(f"Total recommendations generated: {len(all_recommendations)}")
else:
    print("No recommendations generated.") 

# Save to Excel
eval_41_df = pd.DataFrame({
    'Precision': [p41],
    'Recall': [r41],
    'F1': [f41],
    'AverageScore': [avg_score_41 ]
})
eval_41_df.to_excel(tabs_path / f"evaluation_metrics_4_1_{timestamp}.xlsx", index=False)

print("\n========== 4.1 METRICS ==========")
print(eval_41_df)




# 4.2: s kontekstom (po filtriranju)
##############################################################
p42, r42, f42 = erst.evaluate_recommender_metrics(
    D_triplets,
    all_final_recommendations,    # these are context-filtered recommendations
    top_n_groundtruth=top_n_groundtruth,
    k_eval=top_k_final_recommendations
)

print(f"final_recs: {len(all_final_recommendations)}")
if all_final_recommendations:
    avg_score_42 = sum(rec[-1] for rec in all_final_recommendations) / len(all_final_recommendations)
else:
    avg_score_42 = 0.0

# shrani v Excel
eval_42_df = pd.DataFrame({
    'Precision': [p42],
    'Recall': [r42],
    'F1': [f42],
    'AverageScore': [avg_score_42]
})
eval_42_df.to_excel(tabs_path / f"evaluation_metrics_4_2_{timestamp}.xlsx", index=False)

print("\n========== 4.2 METRICS ==========")
print(eval_42_df)

#%% STEP 19: PRINT SAMPLES FOR DEBUGGING
##==================================================================================
print("========== STEP 19: PRINT SAMPLES FOR DEBUGGING ==========")

"""
Shows sample recommendations, data structures, and types to validate logic.
"""

print("\n========== SAMPLE RECOMMENDATION ==========")
if top20_recs:
    print(f"First sample recommendation:\n{top20_recs[0]}")
else:
    print("No sample recommendations available.")


print(f"\n========== DETAILS FOR USER {uID} ==========")
print(f"Number of recommendations for user {uID}: {len(top20_recs)}")


print("Ground truth example from D_lst:")
for row in D_triplets[:3]:
    print(f"user: {row[0]}, action: {row[1]}, type(action): {type(row[1])}")

print("Recommendation example from all_recommendations:")
for row in all_recommendations[:3]:
    print(f"user: {row[0]}, action: {row[1]}, type(action): {type(row[1])}")

# %%
