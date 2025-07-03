#%% Import required libraries
##==================================================================================
import numpy as np
import cProfile
import seaborn as sns
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBasic, BaselineOnly, NMF, accuracy
from sklearn.model_selection import ShuffleSplit
from surprise.model_selection import train_test_split, cross_validate, KFold
from collections import defaultdict
from pathlib import Path
import datetime
import pickle
import elderly_recsys_tools as erst

#%% Define file paths for input data and output results
##==================================================================================

# Get base directory
BASE_DIR = Path(__file__).resolve().parents[1]  

data_path = BASE_DIR / 'data'
figs_path = BASE_DIR / 'latex' / 'figs'
tabs_path = BASE_DIR / 'latex' / 'tabs'

#%% Load data from Excel files
##==================================================================================

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

# %% Create sets and dicts
##==================================================================================

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
#aspect_groups_lst = ['activity', 'phy_health', 'ment_helath', 'soc_helath']
#score_groups_lst = ['activity', 'menHealOsn', 'menHealVisje2', 'menHealVisje4', 'phyHealNacin', 'phyHealOrganski']
uID_scores_dc = {}
uID_scores_dc['activity'] = uID_activity_scores_dc
uID_scores_dc['menHealOsn'] = uID_menHealOsn_scores_dc
uID_scores_dc['menHealVisje2'] = uID_menHealVisje2_scores_dc
uID_scores_dc['menHealVisje4'] = uID_menHealVisje4_scores_dc
uID_scores_dc['phyHealNacin'] = uID_phyHealNacin_scores_dc
uID_scores_dc['phyHealOrganski'] = uID_phyHealOrganski_scores_dc

#%% Settings
##==================================================================================

# Select which aspect groups to include in the recommendation process
# Available options include: 'activity', 'phy_health', 'ment_helath', 'soc_helath'

# Here, only the 'activity' aspect is used for simplicity
aspect_groups_lst = ['activity'] #, 'phy_health'] #, 'ment_helath', 'soc_helath']

# Display the activity score dictionary for inspection (optional line)
# uID_scores_dc['activity']

#%% Set lists of users and actions accoring to selected aspect 
##==================================================================================

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

#%% Load contexts
##==================================================================================

# Reload the custom module in case it was modified (useful during development)
importlib.reload(erst)

# Generate a new column 'kontekst' by combining valid time context values (C_T1, C_T2, C_T3)
# The `get_context` function handles this by filtering and formatting time descriptions
activityContextGen_df['kontekst'] = activityContextGen_df.apply(lambda row: erst.get_context(row['act_C_T1'], row['act_C_T2'], row['act_C_T3']), axis=1)
# print(df[['Single_action', 'C_T1', 'C_T2', 'C_T3', 'kontekst']])

# Extract the context strings into a list for further use
context_lst = activityContextGen_df['kontekst'].tolist()

# Remove the last 6 elements (likely placeholders or incomplete entries)
context_lst = context_lst[:-6]

#%% Generate sequence of actions
##==================================================================================

importlib.reload(erst)

# Define the maximum allowed length of action sequences
act_max_len = 2

# Generate all possible sequences of actions (represented by actIDs)
# using the custom function `get_list_of_actions`. This function likely returns
# combinations/permutations of actions from the provided list (up to the specified length).
seq_actID_lst = erst.get_list_of_actions(single_actID_lst, act_max_len)

#%%
# @brief compute data matrix
"""
Assumptions:
- each action belongs to a question => higher anwser to this question
  means higher relevance to this activity for this user
- each user has his score for each question and higher score to
  this question means higher relevance to this group of activities (=question)
- relevance r = score * anwser 
- if obtained r is positive, we add it to the matrix
"""

# Settings for limiting the number of users and actions considered in the matrix.
uIDs_n, acts_n = 100, 80
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n]
#uIDsIn, seq_actID_lstIn = uIDs, seq_actID_lst

# Extract only single-action sequences for relevance computation.
actID_lstIn = [act[0] for act in seq_actID_lstIn if len(act)==1]

# Method for relevance computation and threshold for inclusion.
meth_code = 'score'
r_T = 0.3 # Threshold

# === Precomputation steps ===

# Precomputed data frames
# Scores for each action and user based on their responses and associated scores.
actID_score_df = erst.get_actID_score_df(uIDsIn, actID_lstIn, actID_qID_dc, uID_scores_dc, all_answers_df, aspect_groups_lst, meth_code)

# Compatibility data between pairs of actions, which likely evaluates how well different actions fit together.
actID_compat_df = erst.get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)

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

#%% Test with plots
## ====================================================================================================

# Visualize the user responses to each question (alternative: uncomment for question-level heatmap)
# sns.heatmap(uID_qID_answers_df)

# Visualize the user responses to each action (rows = users, columns = actions)
# Each cell shows how relevant a given action is to a specific user based on their original response to the associated question
sns.heatmap(uID_actID_answers_df)

#%% Data matrix - D_lst
## ====================================================================================================

"""
Generate the final data matrix D_lst, where each element is a tuple:
(user_id, action_sequence, relevance_score)
The function computes relevance scores based on:
- user responses to questions (uID_actID_answers_df),
- precomputed action scores (actID_score_df),
- and optionally compatibility between actions (actID_compat_df).
"""

# # Only sequences with a relevance score above the defined threshold (r_T) are included.
# D_lst = erst.get_dataMat(uIDsIn, seq_actID_lstIn, uID_actID_answers_df, actID_score_df, actID_compat_df, r_T, meth_code)

# Mentor's new and laready built D_lst matrix: 
with open(data_path / 'D_contx_a3_sparse_mat.pkl', "rb") as fp:
    D_lst = pickle.load(fp)

D_lst = D_lst[:10000]  # vzameš le prvih 1000 zapisov -- za namene hitrega testiranja 

ratings = [x[3] for x in D_lst]

print("\n========== D_lst LOADED ==========")
print(f"Total rows in D_lst: {len(D_lst)}")
print(f"Unique users: {len(set(x[0] for x in D_lst))}")
print(f"Unique actions: {len(set(x[1] for x in D_lst))}")
print(f"Min rating: {min(ratings):.3f}, Max rating: {max(ratings):.3f}, Mean: {np.mean(ratings):.3f}")

#%% Random context generator
## ====================================================================================================

""""
Test
This block tests whether a randomly generated context is feasible for a given action.
"""

importlib.reload(erst)

# Step 1: Collect all possible contexts corresponding to the selected actions
all_contexts = [actID_context_dc[actID] for actID in actID_lstIn]

# Step 2: Generate a random context from the list of available contexts
cntx = erst.get_random_context(all_contexts)

# Step 3: Choose a specific action ID for testing feasibility
actID = 'Ac_AB4_8_Act04'

# Step 4: Check if the randomly generated context is feasible (i.e., valid) for the selected action
erst.is_action_context_feasibleQ(actID, cntx, actID_context_dc)

#%% Evaluation steps
## ===================================================================================================

"""
1. Generate data matrix
2. Use Matrix factorisation to obtain full D. 
3. Define test uIDs
4. For uID in uIDs
  - generate (select) contexts for uID
  - generate recommnedation: ((a_i:i):j) = argmax (D(uID, :))
  - filter out those compatible with the context
  - recommend first three sequences: parametric form
  - recommend first three sequences: textual form
"""

#%% Export recommendations and corresponding contextual explanations
## ==========================================================================================
# STEP-BY-STEP PIPELINE: Load Data → Tune → Cross-Validate → Train Final Model

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

importlib.reload(erst)

# Settings - Omeji podatke za hitrejše testiranje
# uIDs_n, acts_n = 100, 80 # Number of users and actions to consider
uIDs_n, acts_n = 20, 20 # Number of users and actions to consider #users cca 600 # 
M = 3 # Number of latent features for matrix factorization (dummy value here)
m = 5 # Number of top recommended actions per user #20
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n] # Subset of users and actions



# ==========================================================
# LOAD CONTEXTUAL DATA AND MATRIX
# ==========================================================

# Mentor's new and laready built D_lst matrix: 
with open(data_path/ 'D_contx_a3_sparse_mat.pkl', "rb") as fp:
    D_lst = pickle.load(fp)

D_lst = D_lst[:10000]  # vzameš le prvih 1000 zapisov -- za namene hitrega testiranja 

# Load context data from the specified file
# This function reads the context definitions and returns a dictionary mapping action IDs to their context data
actID_context_dc = erst.load_context_data(activityContextGen_df)

# Define an example context for generating recommendations 
context = {'C_T': 'ob kosilu', 'C_P': 'kjerkoli'}



# ==========================================================
# PREPARE DATA FOR SURPRISE
# ==========================================================

df = pd.DataFrame(D_lst, columns=['user_id', 'item_id', 'context', 'rating'])

# Odstranimo 'context', ker ga Surprise ne podpira:
df_for_surprise = df[['user_id', 'item_id', 'rating']]
print("\n========== PREPARED DATAFRAME ==========")
print(f"Shape of df_for_surprise: {df_for_surprise.shape}")

# Prepare data
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df_for_surprise, reader)



# ==========================================================
# ALGORITHM COMPARISON (SVD, KNN, BASELINE, NMF)
# ==========================================================

algorithms = {
    "SVD": SVD(),
    "KNNBasic": KNNBasic(),
    "BaselineOnly": BaselineOnly(),
    "NMF": NMF()
}
metrics = defaultdict(list)
kf = KFold(n_splits=3, random_state=42, shuffle=True)

for name, algo in algorithms.items():
    print(f"\n=== Evaluating {name} ===")
    for fold, (trainset, testset) in enumerate(kf.split(data)):
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae  = accuracy.mae(predictions, verbose=False)
        mse  = accuracy.mse(predictions, verbose=False)
        fcp  = accuracy.fcp(predictions)
        metrics['algorithm'].append(name)
        metrics['fold'].append(fold+1)
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['mse'].append(mse)
        metrics['fcp'].append(fcp)

summary_df = pd.DataFrame(metrics).groupby("algorithm").agg({
    "rmse":"mean",
    "mae":"mean",
    "mse":"mean",
    "fcp":"mean"
}).reset_index()
print("\n========== ALGORITHM COMPARISON SUMMARY ==========")
print(summary_df)
summary_df.to_excel(tabs_path / "algorithm_comparison.xlsx", index=False)
plt.figure(figsize=(8,5))
plt.bar(summary_df["algorithm"], summary_df["rmse"], color="lightblue")
plt.ylabel("Average RMSE")
plt.title("Povprečni RMSE za algoritme")
plt.tight_layout()
plt.show()



# ==========================================================
# SVD TUNING + AND GRID SEARCH
# ==========================================================

# # Define hyperparameter search space for the SVD algorithm (from Surprise library)
# param_grid = {
#     'n_factors': [50, 100, 150], # Number of latent factors
#     'n_epochs': [10, 20, 30], # Number of training epochs
#     'lr_all': [0.002, 0.005], # Learning rate
#     'reg_all': [0.02, 0.1] # Regularization term
# }

param_grid = {
    'n_factors': [10],      # namesto 50–150
    'n_epochs': [5],        # namesto 10–30
    'lr_all': [0.002],      # pusti eno vrednost
    'reg_all': [0.02]
}

profile = cProfile.Profile() 
profile.enable() 

# Run grid search to find the best combination of hyperparameters
gs = erst.grid_search(data, 'SVD', param_grid)
best_params = gs.best_params['rmse']
print("\n========== GRID SEARCH COMPLETED ==========")
print(f"Best params: {best_params}")
print("Best RMSE score:", gs.best_score['rmse'])

profile.disable()
# profile.print_stats(sort='cumtime')  



# ==========================================================
# CROSS-VALIDATION AND METRIC EVALUATION
# ==========================================================

profile = cProfile.Profile()
profile.enable()
cv_results_df = erst.perform_cross_validation(
    data=data, 
    model_class=lambda: SVD(**best_params),  # tovarna za nov model
    n_splits=2, 
    random_state=42
)
profile.disable()
# profile.print_stats(sort='cumtime')
print("\n========== CROSS-VALIDATION RESULTS ==========")
print(cv_results_df)

# Izvlečemo srednje vrednosti iz vrstice 'Mean'
mean_metrics = cv_results_df[cv_results_df['Fold'] == 'Mean'].iloc[0]

# Pripravimo seznam za prikaz
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

print("\nAverage metrics across folds:")
print(results_list)



# ==========================================================
# INITIALIZE MODEL / TRAIN FINAL MODEL ON FULL DATASET OR SPLIT DATASET
# ==========================================================

# # Train the model on the full training set
# trainset = data.build_full_trainset()

# Train the model on the training set
trainset, testset = train_test_split(data, test_size=0.2)
print("\n========== TRAINSET INFO ==========")
print(f"Number of users in trainset: {trainset.n_users}")
print(f"Number of items in trainset: {trainset.n_items}")
print(f"Testset length: {len(testset)}")

# Inicializacija modela z najboljšimi parametri
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

# Izračun osnovnih metrik na testnem delu
print("\n========== TEST SET EVALUATION ==========")
print(f"RMSE: {accuracy.rmse(predictions)}")
print(f"MAE:  {accuracy.mae(predictions)}")
print(f"MSE:  {accuracy.mse(predictions)}")
print(f"FCP:  {accuracy.fcp(predictions)}")

print("\n========== MODEL SUMMARY ==========")
print(f"Number of users in trainset: {trainset.n_users}")
print(f"Number of items in trainset: {trainset.n_items}")
print(f"Total unique actions in D_lst: {len(set(x[1] for x in D_lst))}")



# ==========================================================
# EXTRACT USER AND ITEM FACTORS (P, Q)
# ==========================================================

# Build user and item factor dicts
user_factors = {trainset.to_raw_uid(uid): model.pu[uid] for uid in range(trainset.n_users)}
item_factors = {trainset.to_raw_iid(iid): model.qi[iid] for iid in range(trainset.n_items)}
print("\n========== FACTORS INFO ==========")
print(f"User factors: {len(user_factors)} extracted")
print(f"Item factors: {len(item_factors)} extracted")



# ===========================================================
# GENERATE EXPLAINABLE RECOMMENDATIONS FOR EACH USER
# ===========================================================

dc_lst_41 = []   # za 4.1
dc_lst_42 = []   # za 4.2
all_recommendations = []           # top-20 brez konteksta
all_final_recommendations = []     # top-5 filtriranih s kontekstom

for uID in uIDsIn:

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
    # best_act_trp_lst = erst.get_recommendations(uID=uID, D_lst=D_lst, n_recommendations=m)
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, n_recommendations=m)
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, context=context, actID_context_dc=actID_context_dc, n_recommendations=m)


    # # GET RECOMMENDATIONS - WITHOUT CONTEXT - USING SURPRISE LIBRARY
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, n_recommendations=m)

    # # GET RECOMMENDATIONS - WITH CONTEXT
    # best_act_trp_lst = erst.get_recommendations(uID=uID, trainset=trainset, model=model, 
    #                 context=context, actID_context_dc=actID_context_dc, n_recommendations=m)


    # najprej 20 kandidatov brez filtriranja
    top20_recs = erst.get_recommendations(
        uID=uID,
        trainset=trainset,
        model=model,
        n_recommendations=20
    )
    # potem filtriraj po kontekstu in vzemi 5
    filtered_recs = [
        rec for rec in top20_recs
        if erst.is_action_context_feasibleQ(rec[1], context, actID_context_dc)
    ]
    final_recs = filtered_recs[:m]


    # all_recommendations.extend(best_act_trp_lst)  # <- Dodamo v glavni seznam

    # Pred tem:
    # all_recommendations.extend(best_act_trp_lst)

    # Namesto tega:
    # all_recommendations.extend([(uID, act_seq, score) for act_seq, score in best_act_trp_lst])

    # all_recommendations.extend([
    #     (uid, act_seq, score) for uid, act_seq, context, score in best_act_trp_lst
    # ])

    all_recommendations.extend(top20_recs)
    all_final_recommendations.extend(final_recs)

    # print(f"\n========== TOP RECOMMENDATIONS FOR USER {uID} ==========")
    # for rec in best_act_trp_lst[:3]:  # da jih ne zasuje
    #     print(f"Action: {rec[1]}, Score: {rec[-1]:.3f}")

    for uid, act_id, score in top20_recs:
        try:
            c_MF_Q = item_factors[act_id]
        except KeyError:
            c_MF_Q = [0.0] * M

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
    
    # za poglavje 4.2 (s kontekstom)
    for uid, act_id, score in final_recs:
        try:
            c_MF_Q = item_factors[act_id]
        except KeyError:
            c_MF_Q = [0.0] * M
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

# %%
# ===========================================================
# EXPORT PRIPOROČIL V EXCEL
# ===========================================================

rec_X_df_41 = pd.DataFrame(dc_lst_41)
rec_X_df_41.to_excel(tabs_path / 'recommendations_4_1.xlsx', index=False)

rec_X_df_42 = pd.DataFrame(dc_lst_42)
rec_X_df_42.to_excel(tabs_path / 'recommendations_4_2.xlsx', index=False)

print(f"Exported {len(rec_X_df_41)} rows for 4.1 recommendations to recommendations_4_1.xlsx")
print(f"Exported {len(rec_X_df_42)} rows for 4.2 recommendations to recommendations_4_2.xlsx")



# ===========================================================
# EVALUATION METRICS
# ===========================================================

# Pretvori D_lst iz četverčkov v trojčke, ki jih funkcija pričakuje --> to damo sam za hitro testiranje brez kontesta
D_triplets = [(uid, iid, rating) for uid, iid, context, rating in D_lst]


# 4.1: brez konteksta (standardna matrika)
##############################################################
p41, r41, f41 = erst.evaluate_recommender_metrics(
    D_triplets,
    all_recommendations,   # to so tvoje brez-kontekstne priporočila
    top_n_groundtruth=5,
    k_eval=m
)

if all_recommendations:
    avg_score_41  = sum(rec[-1] for rec in all_recommendations) / len(all_recommendations)
    print(f"Average score: {avg_score_41 :.3f}")
    print(f"Total recommendations generated: {len(all_recommendations)}")
else:
    print("No recommendations generated.") 

# shrani v Excel
eval_41_df = pd.DataFrame({
    'Precision': [p41],
    'Recall': [r41],
    'F1': [f41],
    'AverageScore': [avg_score_41 ]
})
eval_41_df.to_excel(tabs_path / 'evaluation_metrics_4_1.xlsx', index=False)

print("\n========== 4.1 METRICS ==========")
print(eval_41_df)




# 4.2: s kontekstom (po filtriranju)
##############################################################
p42, r42, f42 = erst.evaluate_recommender_metrics(
    D_triplets,
    all_final_recommendations,    # to so kontekst filtrirana priporočila
    top_n_groundtruth=5,
    k_eval=m
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
eval_42_df.to_excel(tabs_path / 'evaluation_metrics_4_2.xlsx', index=False)

print("\n========== 4.2 METRICS ==========")
print(eval_42_df)







print("\n========== SAMPLE RECOMMENDATION ==========")
if top20_recs:
    print(f"First sample recommendation:\n{top20_recs[0]}")
else:
    print("No sample recommendations available.")


print(f"\n========== DETAILS FOR USER {uID} ==========")
print(f"Number of recommendations for user {uID}: {len(top20_recs)}")

# %%
