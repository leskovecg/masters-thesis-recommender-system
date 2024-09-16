#%% Generate data matrix
import numpy as np
import pandas as pd
import scipy.sparse as ss
import pickle
import time
import cProfile
import pstats
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import importlib
import random
import elderly_recsys_tools as erst

# Set the threshold for time measurements
TIME_THRESHOLD = 10  # 10 milliseconds

profile = cProfile.Profile()
profile.enable()

#%% Settings & load data
data_path = 'Data/'
figs_path = 'Figs/'
tabs_path = 'Tabs/'

#%%
activities_df = pd.read_excel(data_path + 'Activities.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
activities_df.rename(columns={nm:'Ac_' + nm for nm in activities_df.columns}, inplace=True)

mentalHealth_df = pd.read_excel(data_path + 'MentalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
mentalHealth_df.rename(columns={nm:'Mh_' + nm for nm in mentalHealth_df.columns}, inplace=True)

physicalHealth_df = pd.read_excel(data_path + 'PhysicalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
physicalHealth_df.rename(columns={nm:'Ph_' + nm for nm in physicalHealth_df.columns}, inplace=True)

socialHealth_df = pd.read_excel(data_path + 'SocialHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
socialHealth_df.rename(columns={nm:'Sh_' + nm for nm in socialHealth_df.columns}, inplace=True)


activityContextGen_df = pd.read_excel(data_path + 'ActivityContextGen_v06.xlsx', sheet_name='ActionLst').replace(-1, np.nan)
scores_and_wgt_df = pd.read_excel(data_path + 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')
#scores_and_wgt_2_df = pd.read_excel(data_path + 'ml_data_with_scores_and_wgt_3_values.xlsx', sheet_name='scores_and_wgt_3_v', header=1)
#wgt_results_annotations_df = pd.read_excel(data_path + 'wgt_results_annotations_3009_4users.xlsx', sheet_name='wgt_results_annotations_3009_4u', header=1).drop(columns=['Unnamed: 0', 'Column1'])
all_answers_df = activities_df.join(mentalHealth_df).join(physicalHealth_df, rsuffix='_r').join(socialHealth_df, rsuffix='_r')




# %% Create sets and dicts
uIDs = list(np.sort(scores_and_wgt_df.index))
activity_qs = ['Ac_AB4_1', 'Ac_AB4_2', 'Ac_AB4_3', 'Ac_AB4_4', 'Ac_AB4_5', 'Ac_AB4_8'] 
#'activity' = [AB4_1, AB4_2, AB4_3, AB4_4, AB4_5, AB4_8]
phy_health_qs = ['Ph_AB1_7', 'Ph_AB1_11', 'Ph_AB3', 'Ph_AB6_1', 'Ph_AB6_5', 'Ph_AB7_1', 'Ph_AB7_5', 'Ph_AB4_2', 'Ph_AB4_3', 'Ph_AB4_4']
#'pyhisicalHealth ' = [AB1_11, AB3, AB6_1, AB6_5, AB7_1, AB7_5, AB4_2, AB4_3, AB4_4]
ment_helath_qs = ['Mh_A75_2', 'Mh_A75_3', 'Mh_A75_4', 'Mh_A75_5', 'Mh_AB1_14', 'Mh_A82_r1', 'Mh_A82_r3']
# 'mental_health' = [A75_2, A75_3, A75_4, A75_5, AB1_14, A82_r1, A82_r3, A83_r]
soc_helath_qs = ['Sh_A83_r', 'Sh_sh_AB98_da_ne']
# 
group_qLst = {'activity':activity_qs,
              'phy_health':phy_health_qs,
              'ment_helath':ment_helath_qs,
              'soc_helath':soc_helath_qs}

group_factor_dc = {'Activities': 'F2', 
                   'PhysicalHealth-organskiSistemi': 'F1',
                   'PhysicalHealth-nacinZivljenja': 'F1',
                   'MentalHealth-osnovno': 'F1',
                   'MentalHealth-visje': 'F3'}


# Create dictionaries to both ways
actID_singleAct_dc = dict(zip(activityContextGen_df['actID'], activityContextGen_df['Single_action']))
singleAct_actID_dc = dict(zip(activityContextGen_df['Single_action'], activityContextGen_df['actID']))



curr_qID = np.nan
curr_qIDs = []
qID_qtxt_dc = {}
qID_singleAct_dc = {}
qID_actID_dc = {}
actID_context_dc = {}
actID_props_dc = {}
qID_Group_dc = {}
singleAct_qID_dc = {}
actID_qID_dc = {}
for ind, row in activityContextGen_df.iterrows():
    
    # New qID
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
        single_act = row['Single_action']
        actID = row['actID']
        qID_singleAct_dc[curr_qID].append(single_act)
        #singleAct_qID_dc[curr_qID+'_'+str(single_act)] = curr_qID
        singleAct_qID_dc.update({curr_qID+'_'+str(single_act):curr_qID})
        qID_actID_dc[curr_qID].append(actID)
        actID_qID_dc.update({str(actID):curr_qID})

        # Action props
        actID_props_dc[actID] = {}
        actID_props_dc[actID]['qID'] = last_qID
        actID_props_dc[actID]['action_prop_1'] = row['action_prop_1']
        actID_props_dc[actID]['action_prop_2'] = row['action_prop_2']
        actID_props_dc[actID]['action_prop_3'] = row['action_prop_3']

        # Context
        actID_context_dc[actID] = {}
        actID_context_dc[actID]['qID'] = last_qID
        actID_context_dc[actID]['C_T1'] = row['C_T1']
        actID_context_dc[actID]['C_T2'] = row['C_T2']
        actID_context_dc[actID]['C_T3'] = row['C_T3']
        actID_context_dc[actID]['C_P1'] = row['C_P1']
        actID_context_dc[actID]['C_P2'] = row['C_P2']
        actID_context_dc[actID]['C_P3'] = row['C_P3']


# uIDs to socres
uID_activity_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['a_F2']))
uID_menHealOsn_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_osnovno_F1']))
uID_menHealVisje2_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F2']))
uID_menHealVisje4_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F4']))
uID_phyHealNacin_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_nacinZivljenja_F1']))
uID_phyHealOrganski_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_organskiSistemi_F1']))

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
aspect_groups_lst = ['activity'] #, 'phy_health'] #, 'ment_helath', 'soc_helath']

uID_scores_dc['activity']

#%% Set lists of users and actions accoring to selected aspect =====================
#single_act_lst = [g for g in singleAct_qID_dc]
single_act_lst = []
for group in aspect_groups_lst:
    for act in singleAct_qID_dc:
        for qID in group_qLst[group]:
            if qID in act:
                single_act_lst.append(act)


#single_actID_lst = [g for g in actID_qID_dc]
single_actID_lst = []
for group in aspect_groups_lst:
    for actID in actID_qID_dc:
        for qID in group_qLst[group]:
            if qID in actID:
                single_actID_lst.append(actID)


#%% ####################################################################################
# Lodad contexts

importlib.reload(erst)

# df = pd.read_excel('C:\\Users\\Gasper\\OneDrive\\FAKS\\MAGISTERIJ\\letnik_2\\SEMESTER_2\\magistrska\\ElderlyActivityContextGen_v01.xlsx')  
# df = pd.read_excel('C:\\Users\\Gasper\\OneDrive\\FAKS\\MAGISTERIJ\\letnik_2\\SEMESTER_2\\magistrska\\ElderlyActivityContextGen_v01.xlsx')  
activityContextGen_df = pd.read_excel(data_path + 'ActivityContextGen_v06.xlsx', sheet_name='ActionLst').replace(-1, np.nan)
activityContextGen_df['kontekst'] = activityContextGen_df.apply(lambda row: erst.get_context(row['C_T1'], row['C_T2'], row['C_T3']), axis=1)
# print(df[['Single_action', 'C_T1', 'C_T2', 'C_T3', 'kontekst']])
context_lst = activityContextGen_df['kontekst'].tolist()
# Now cut the last 6 elements
context_lst = context_lst[:-6]



#%% ####################################################################################
# Generate sequence of actions
#importlib.reload(erst)
act_max_len = 2
seq_actID_lst = erst.get_list_of_actions(single_actID_lst, act_max_len)

#%%
# @brief compute data matrix
# Assumptions:
# - each action belongs to a question => higher anwser to this questio n
#   means higher relevance to this activity for this user
# - each user has his score for each question and higher score to
#   this question means higher relevance to this group of activities (=question)
# - relevance r = score * anwser 
# - if obtained r is positive, we add it to the matrix
#   

# Settings
uIDs_n, acts_n = 100, 80
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n]
#uIDsIn, seq_actID_lstIn = uIDs, seq_actID_lst

actID_lstIn = [act[0] for act in seq_actID_lstIn if len(act)==1]
meth_code = 'score'
r_T = 0.3 # Threshold

# Precomputed data frames
actID_score_df = erst.get_actID_score_df(uIDsIn, actID_lstIn, actID_qID_dc, uID_scores_dc, all_answers_df, aspect_groups_lst, meth_code)
actID_compat_df = erst.get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)
uID_qID_answers_df = erst.get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst)

uID_actID_answers_df = pd.DataFrame(index=uID_qID_answers_df.index)
for actID in actID_lstIn:
    if actID_qID_dc[actID] in uID_qID_answers_df:
        uID_actID_answers_df[actID] = uID_qID_answers_df[actID_qID_dc[actID]]
    else:
        print ('Error:' + actID)


#%% Test with plots
#sns.heatmap(uID_qID_answers_df)
#sns.heatmap(uID_actID_answers_df)


#%% Data matrix
D_lst = erst.get_dataMat(uIDsIn, seq_actID_lstIn, uID_actID_answers_df, actID_score_df, actID_compat_df, r_T, meth_code)
#D_df = get_dataMat(uIDs, seq_act_lst, meth_code)
#D_df.to_excel(data_path + meth_code+'_D_df.xlsx')


#%% ====================================================================================================
# Random context generator

# @brief Get random context 
def get_one_random_context(full_cntx):
    
    C_T = random.choice(['C_T1', 'C_T2', 'C_T3'])
    C_P = random.choice(['C_P1', 'C_P2', 'C_P3'])

    c_cntx = {'qID': full_cntx['qID'], 'C_T': full_cntx[C_T], 'C_P': full_cntx[C_P], 'C_A':''}
    
    return c_cntx


# @brief Get random context 
def get_random_context(all_contexts):
    
    full_cntx = random.choice(all_contexts)
    C_T = random.choice(['C_T1', 'C_T2', 'C_T3'])
    C_P = random.choice(['C_P1', 'C_P2', 'C_P3'])

    c_cntx = {'qID': full_cntx['qID'], 'C_T': full_cntx[C_T], 'C_P': full_cntx[C_P], 'C_A':''}
    
    return c_cntx

# @brief test if a given action fits to a given context
def is_action_context_feasibleQ(actID, cntx, actID_context_dc):

    f_cntx = actID_context_dc[actID] # Full context
    f_cntx = actID_context_dc[actID]
    C_Ts = [f_cntx[k].strip() for k in ['C_T1', 'C_T2', 'C_T3'] if isinstance(f_cntx[k], str)]
    C_Ps = [f_cntx[k].strip() for k in ['C_P1', 'C_P2', 'C_P3'] if isinstance(f_cntx[k], str)]

    if (cntx['C_T'] in C_Ts) and (cntx['C_P'] in C_Ps):
        return True 
    else:
        return False

# Test
all_contexts = [actID_context_dc[actID] for actID in actID_lstIn]
cntx = get_random_context(all_contexts)
actID = 'Ac_AB4_8_Act04'
is_action_context_feasibleQ(actID, cntx, actID_context_dc)




#%% ===================================================================================================
# Evaluation steps
# 1. Generate data matrix
# 2. Use Matrix factorisation to obtain full D. 
# 3. Define test uIDs
# 4. For uID in uIDs
#   - generate (select) contexts for uID
#   - generate recommnedation: ((a_i:i):j) = argmax (D(uID, :))
#   - filter out those compatible with the context
#   - recommend first three sequences: parametric form
#   - recommend first three sequences: textual form

#@brief returns m best actions triples
def get_recommendations(uID, D_lst, m):

    c_act_trp_lst = [x for x in D_lst if x[0]==uID] # Select all actions of this user
    mbest_act_trp_lst = sorted(c_act_trp_lst, key=lambda x: x[2], reverse=True) # Sort it
    
    return mbest_act_trp_lst[:m]



#%% ==========================================================================================
# Export recommendations and contexts
# What to export:
#  - uID
#  - one recommendation in textual form, uIDs can repeat, so uID is not the 
#   index of the DataFrame
#  - explanations are based on:
#    - Context: ad context
#    - uID anwser for this question: add his anwsers regarding actID 
#    - scores: ad scores
#    - explainable AI: P and Q vectors
# 
# In this way, the explanations will be reprashed from:
#   - context: it is a good weather, right time, ..... [add weather?]
#   - qaID: you appretiate it a lot
#   - scores: you are good at physical activitiy 
#   - P and Q: segment them and find groups 
#   



uIDs_n, acts_n = 100, 80
M = 3 # Number of components in MF
m = 4 # Number of recommended actions
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n]


dc_lst = []
for uID in uIDsIn:

    # get qaIDs
    c_anws_a = dict(uID_qID_answers_df.loc[uID,:])
    c_anws_txt = {qID:qID_qtxt_dc[qID] for qID in c_anws_a}

    # get scores
    c_score = uID_scores_dc['activity'][uID]


    # get P and Q
    c_MF_P = [1,2,3]
    c_MF_Q = [3,2,1]


    # Get recommnedations
    best_act_trp_lst = get_recommendations(uID, D_lst, m)

    for act_trp in best_act_trp_lst:

        c_acts = act_trp[1] 

        # Get action props
        for c_act in c_acts: # We go to single action explanation
            c_act_c = c_act
            c_act_txt = actID_singleAct_dc[c_act]
            c_act_prop = actID_props_dc[c_act]
        
            # Get one possible context for this recommendation
            full_cntx = actID_context_dc[c_act]
            c_cntx = get_one_random_context(full_cntx)

            # Store it all
            c_dc = {'uID': uID, 'Act_c:':c_act_c, 'Act_txt':c_act_txt, 'Act_prop': c_act_prop, 'Cntx': c_cntx, 'Anws_a': c_anws_a, 'Anws_txt':c_anws_txt, 'Score':c_score, 'MF_P':c_MF_P, 'MF_Q':c_MF_Q}
            dc_lst.append(c_dc)


rec_X_df = pd.DataFrame(dc_lst)

# Store 
rec_X_df.to_excel(tabs_path + 'recom_acts_sample.xlsx')































#%% ===========================================================================================================
dc_lst = []
ver1_dc = {'Ver': 1, 'P':0.3, 'R':0.6, 'F':0.45}
dc_lst.append(ver1_dc)
ver2_dc = {'Ver': 1, 'P':0.3, 'R':0.6, 'F':0.45}
dc_lst.append(ver2_dc)
recSys_succ_df = pd.DataFrame(dc_lst)
recSys_succ_df.to_excel(tabs_path + 'recSys_succ_df.xlsx')
recSys_succ_df.to_latex(tabs_path + 'recSys_succ_df.tex')


#%% Small tests
uIDsIn, seq_act_lstIn = uIDs[:5], seq_act_lst[:5]
#meth_code = 'score'
#actID_lstIn = list(actID_qID_dc.keys())[:5]
#actID_score_df = erst.get_actID_score_df(uIDsIn, actID_lstIn, actID_qID_dc, uID_scores_dc, all_answers_df, group_str, meth_code)
#print(actID_score_df)
#compat_df = erst.get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)
#print(comp_df)
#uID, act_seq = uIDs[0], seq_act_lst[-1]
#c_r = erst.get_score_estimation(uID, act_seq, actID_score_df, actID_compat_df, meth_code)


#%%
# Stats
# D \in [696, 232]
# D elts = 161472
# Non nans = 80736, that is 50%
##################################################################
# Check if D_lst exists in saved form
# try:
#     with open("data_df.pkl", "rb") as f:
#         data_df = pickle.load(f)
#     print("Loaded data_df from pickle.")
# except:

# Settings
uIDs_n, acts_n = 100, 80
uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n]
#uIDsIn, seq_actID_lstIn = uIDs, seq_actID_lst

actID_lstIn = [act[0] for act in seq_actID_lstIn if len(act)==1]
meth_code = 'score'
r_T = 0.3 # Threshold

#start_time = time.time()  # Start time
#D_lst = erst.get_dataMat(uIDsIn, seq_actID_lstIn, uID_actID_answers_df, actID_score_df, actID_compat_df, r_T, meth_code)
#end_time = time.time()  # End time
#elapsed_time = end_time - start_time  # Calculate elapsed time
#data_df_context = pd.DataFrame(D_lst, columns=['user', 'item', 'rating','context'])
#print(f"Time taken to get data matrix: {elapsed_time} seconds")

start_time = time.time()  # Start time
D_lst = erst.get_dataMat(uIDsIn, seq_actID_lstIn, uID_actID_answers_df, actID_score_df, actID_compat_df, r_T, meth_code)
end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating'])
    
    # Saving DataFrame to CSV file
    # data_df.to_csv('data_matrix_1.csv', index=False)  # Index set to False to not include row indices in the CSV file
    
    # with open("data_df.pkl", "wb") as f:
    #     pickle.dump(data_df, f)
    # print("Saved data_df to pickle.")
print(f"Time taken to get data matrix: {elapsed_time} seconds")
#################################################################
# %%
'''
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
X_sm =  ss.coo_array((data, (row, col)), dtype=np.int8) # shape=(4, 4))
'''
# %% Load libs
from surprise import Dataset
from surprise import Reader

#%%
# Assuming D_lst contains tuples of (user, item, rating)
D_lst = erst.get_dataMat(uIDs, seq_act_lst, meth_code)

# The columns must correspond to user id, item id and ratings (in that order).
data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating'])

# A reader is needed but only the rating_scale param is required.
reader = Reader(rating_scale=(data_df['rating'].min(), data_df['rating'].max()))
reader_context = Reader(rating_scale=(data_df_context['rating'].min(), data_df['rating'].max()))

#####################################################
if data_df['rating'].isnull().sum() > 0:
    print("There are NaN values in the rating column.")

nan_count = data_df['rating'].isnull().sum()
print(f"Number of NaN values in the rating column: {nan_count}")

data_df = data_df.dropna(subset=['rating'])

if data_df_context['rating'].isnull().sum() > 0:
    print("There are NaN values in the rating column.")

nan_count = data_df_context['rating'].isnull().sum()
print(f"Number of NaN values in the rating column: {nan_count}")

data_df_context = data_df_context.dropna(subset=['rating'])
#####################################################

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(data_df[['user', 'item', 'rating']], reader)
data_context = Dataset.load_from_df(data_df_context[['user', 'item', 'rating', 'context']], reader_context)
# %%
from surprise import SVD, KNNBasic, NMF, accuracy
from surprise.model_selection import train_test_split
import time

def evaluate_algorithm(data, algorithm, sim_options=None):
    
    # Splitting the data into training and testing sets (80%-20% split)
    trainset, testset = train_test_split(data, test_size=0.2)

    # Choose the algorithm based on input
    if algorithm == 'SVD':
        algo = SVD()
    elif algorithm == 'KNNBasic':
        if sim_options is None:
            sim_options = {'name': 'cosine', 'user_based': True}
        algo = KNNBasic(sim_options=sim_options)
    elif algorithm == 'NMF':
        algo = NMF()
    else:
        raise ValueError("Invalid algorithm specified")

    # Training the algorithm
    start_time = time.time()
    algo.fit(trainset)
    elapsed_time = time.time() - start_time

    # Making predictions
    predictions = algo.test(testset)

    # Calculating RMSE and MAE
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Print results
    print(f"RMSE of {algorithm} Algorithm: {rmse}")
    print(f"MAE of {algorithm} Algorithm: {mae}")
    print(f"Time taken to train {algorithm} Algorithm: {elapsed_time} seconds")


evaluate_algorithm(data, 'SVD')
evaluate_algorithm(data, 'KNNBasic', sim_options={'name': 'cosine', 'user_based': True})
evaluate_algorithm(data, 'NMF')

evaluate_algorithm(data_context, 'SVD')
evaluate_algorithm(data_context, 'KNNBasic', sim_options={'name': 'cosine', 'user_based': True})
evaluate_algorithm(data_context, 'NMF')


# %%
def get_dataMat_with_context(uIDs, seq_act_lst, meth_code, context_lst):
    r_T = 0  
    D_lst_with_context = []  

    for i, uID in enumerate(uIDs):
        for j, act_seq in enumerate(seq_act_lst):
            # Get the context for the current user and action sequence
            c = context_lst[i]  # Assuming context_list is aligned with uIDs
            # Get the rating for this action
            raw_r = get_rating_estimation(uID, act_seq, singleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
            # Compatibility score
            comp_score = get_compatibility_score(act_seq)
            
            if raw_r <= r_T:
                r = np.nan
            else:
                r = comp_score * raw_r

            # Append the tuple with context to the list
            D_lst_with_context.append([uID, act_seq, r, c_seq])

    return D_lst_with_context

# Now, when you call this function, you pass the context list as well:
# Assuming context_list is already populated with context strings
#D_lst_with_context = get_dataMat_with_context(uIDs, seq_act_lst, meth_code, context_lst)


import pandas as pd
import numpy as np

def get_dataMat_with_context(uIDs, seq_act_lst, meth_code, context_lst):
    D_lst_with_context = []  

    for uID in uIDs:
        for j, act_seq in enumerate(seq_act_lst):
            context = context_lst[j]  # Access the context using index j
            # Append the required elements to the list
            D_lst_with_context.append([uID, act_seq, meth_code, context])

    # Convert the list to a DataFrame
    df = pd.DataFrame(D_lst_with_context, columns=['user(uIDs)', 'item(seq_act_lst)', 'rating(meth_code)', 'context'])
    return df

# Assuming uIDs, seq_act_lst, meth_code, and context_lst are defined
D_lst_with_context_df = get_dataMat_with_context(uIDs, seq_act_lst, meth_code, context_lst)
D_lst_with_context_df

# %%

def get_dataMat(uIDs, seq_act_lst, meth_code, context_lst):
    
    r_T = 0  # Rating threshold
    D_lst = []

    for uID in uIDs:
        for j, act_seq in enumerate(seq_act_lst):
            # Access the context using index j
            context = context_lst[j]

            # Rating for this action
            raw_r = get_rating_estimation(uID, act_seq, singleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
            # Compatibility score
            comp_score = get_compatibility_score(act_seq)

            if raw_r <= r_T:
                r = np.nan
            else:
                r = comp_score * raw_r

            # Append the tuple including context to the list
            D_lst.append([uID, act_seq, r, context])
    
    return D_lst

# Assuming uIDs, seq_act_lst, meth_code, and context_lst are defined
D_lst_with_context = get_dataMat(uIDs, seq_act_lst, meth_code, context_lst)


start_time = time.time()  # Start time
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code, context_lst)
end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating', 'context'])

# Saving DataFrame to CSV file
data_df.to_csv('data_matrix_2.csv', index=False)  # Index set to False to not include row indices in the CSV file


# %%
def get_dataMat(uIDs, seq_act_lst, meth_code, context_lst = 0):
    
    r_T = 0  # Rating threshold
    D_lst = []

    context_cycle = itertools.cycle(context_lst)

    for uID in uIDs:
        for act_seq in seq_act_lst:
            # Access the context using index j
            context = next(context_cycle)  # Get next context

            # Rating for this action
            raw_r = get_rating_estimation(uID, act_seq, singleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
            # Compatibility score
            comp_score = get_compatibility_score(act_seq)

            if raw_r <= r_T:
                r = np.nan
            else:
                r = comp_score * raw_r

            if context is not 0:
                # Append the tuple including context to the list
                D_lst.append([uID, act_seq, r, context])
            else:
                D_lst.append([uID, act_seq, r])
    
    return D_lst

# Assuming uIDs, seq_act_lst, meth_code, and context_lst are defined
D_lst_with_context = get_dataMat(uIDs, seq_act_lst, meth_code, context_lst)


start_time = time.time()  # Start time
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code, context_lst)
end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating', 'context'])

# Saving DataFrame to CSV file
data_df.to_csv('data_matrix_2.csv', index=False) 
# %%
from surprise import SVD, KNNBasic, NMF, accuracy

unique_contexts = data_df_context['context'].unique()

# %%
context_datasets = {}

for context in unique_contexts:
    context_data = data_df_context[data_df_context['context'] == context]
    context_datasets[context] = context_data[['user', 'item', 'rating']]
# %%
from surprise import Dataset, Reader


def train_for_context(dataframe, algorithm, sim_options=None):
    reader = Reader(rating_scale=(dataframe['rating'].min(), dataframe['rating'].max()))
    data = Dataset.load_from_df(dataframe, reader)
    # trainset = data.build_full_trainset()

    # Splitting the data into training and testing sets (80%-20% split)
    trainset, testset = train_test_split(data, test_size=0.2)

    # Initialize the algorithm
    if algorithm == 'SVD':
        algo = SVD()
    elif algorithm == 'KNNBasic':
        algo = KNNBasic(sim_options=sim_options)
    elif algorithm == 'NMF':
        algo = NMF()
    else:
        raise ValueError("Invalid algorithm specified")

    # Train the algorithm
    algo.fit(trainset)


    # Making predictions
    predictions = algo.test(testset)

    # Calculating RMSE and MAE
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Print results
    print(f"RMSE of {algorithm} Algorithm: {rmse}")
    print(f"MAE of {algorithm} Algorithm: {mae}")
    print(f"Time taken to train {algorithm} Algorithm: {elapsed_time} seconds")


    return algo

models_by_context = {}

for context in unique_contexts:
    print(f"Training model for context: {context}")
    models_by_context[context] = train_for_context(context_datasets[context], 'SVD')


print(f"Training model without context")
train_for_context(data_df, 'SVD')

# %%
current_context = "nd"  # This should be determined based on your application's logic
selected_model = models_by_context.get(current_context)

if not selected_model:
    # Handle cases where there is no model for the current context
    # This could involve using a default model or another fallback strategy
    pass
# %%

testset = [[uID, itemID, 0]]  # 0 is a dummy rating since the actual rating is unknown
predictions = selected_model.test(testset)

# Extract the predicted rating
predicted_rating = predictions[0].est


profile.disable()
profile.print_stats()

stats = pstats.Stats(profile).sort_stats('cumulative')
filtered_stats = [(x[0], x[1], x[2], x[3]) for x in stats.stats.items() if x[3][0] > TIME_THRESHOLD]
stats = pstats.Stats(profile, stream=open('/tmp/output.txt', 'w'))
stats.stats = dict(filtered_stats)
stats.print_stats()

# Optionally, extract profiling data for plotting
profile_data = [(x[2], x[1]) for x in stats.fcn_list]
function_names, total_times = zip(*profile_data)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(function_names, total_times, color='skyblue')
plt.show()



stats = pstats.Stats(profile).sort_stats('cumulative')
stats.print_stats()

# Define a threshold (e.g., 1% of the total time)
threshold = 0.01 * stats.total_tt

# Extracting profiling data
profile_data = [(x[2], x[1]) for x in stats.fcn_list]
function_names, total_times = zip(*profile_data)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(function_names, total_times, color='skyblue')
plt.xlabel('Total Time Spent (seconds)')
plt.title('Time Distribution')
# %%


# Process profiling data
stats = pstats.Stats(profile).sort_stats('cumulative')
filtered_stats = {key: val for key, val in stats.stats.items() if val[0] > TIME_THRESHOLD}
stats.stats = filtered_stats
stats.print_stats()

# Optionally, extract profiling data for plotting
profile_data = [(key, val[0]) for key, val in stats.stats.items()]
function_names, total_times = zip(*profile_data)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(function_names, total_times, color='skyblue')
plt.show()
# %%

# Process profiling data
stats = pstats.Stats(profile).sort_stats('cumulative')
filtered_stats = {key: val for key, val in stats.stats.items() if val[0] > TIME_THRESHOLD}
stats.stats = filtered_stats
stats.print_stats()

# Optionally, extract profiling data for plotting
profile_data = [(key, val[0]) for key, val in stats.stats.items()]

# Check if there is data to plot
if profile_data:
    function_names, total_times = zip(*profile_data)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(function_names, total_times, color='skyblue')
    plt.xlabel('Total Time (seconds)')
    plt.ylabel('Function Names')
    plt.title('Profiling Data')
    plt.show()
else:
    print("No profiling data above the threshold.")

# %%


stats = pstats.Stats(profile).sort_stats('cumulative')
filtered_stats = [(key, val.total_tt) for key, val in stats.stats.items() if val.total_tt > TIME_THRESHOLD]

# Check if there is data to plot
if filtered_stats:
    function_names, total_times = zip(*filtered_stats)

    # Plotting
    plt.figure(figsize=(10, 8))
    y_pos = range(len(function_names))
    plt.barh(y_pos, total_times, color='skyblue')
    plt.yticks(y_pos, function_names)
    plt.xlabel('Total Time (seconds)')
    plt.ylabel('Function Names')
    plt.title('Profiling Data')
    plt.show()
else:
    print("No profiling data above the threshold.")
# %%

# Process profiling data
stats = pstats.Stats(profile).sort_stats('cumulative')
filtered_stats = [(key, value[3]) for key, value in stats.stats.items() if value[3] > TIME_THRESHOLD]

# Check if there is data to plot
if filtered_stats:
    function_names, total_times = zip(*filtered_stats)

    # Plotting
    plt.figure(figsize=(10, 8))
    y_pos = range(len(function_names))
    plt.barh(y_pos, total_times, color='skyblue')
    plt.yticks(y_pos, function_names)
    plt.xlabel('Total Time (seconds)')
    plt.ylabel('Function Names')
    plt.title('Profiling Data')
    plt.show()
else:
    print("No profiling data above the threshold.")
# %%

from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, KNNBasic, NMF, accuracy
from surprise.model_selection import PredefinedKFold
import time

def train_for_context(dataframe, algorithm, sim_options=None):
    reader = Reader(rating_scale=(dataframe['rating'].min(), dataframe['rating'].max()))

    # Splitting the data into training and testing sets (80%-20% split)
    train_df, test_df = train_test_split(dataframe, test_size=0.2)

    # Converting the splits into surprise Dataset format
    train_data = Dataset.load_from_df(train_df, reader)
    test_data = Dataset.load_from_df(test_df, reader)

    # Using PredefinedKFold with the predefined splits
    pkf = PredefinedKFold()
    trainset = train_data.construct_trainset(train_data.raw_ratings)
    testset = test_data.construct_testset(test_data.raw_ratings)

    # Initialize the algorithm
    if algorithm == 'SVD':
        algo = SVD()
    elif algorithm == 'KNNBasic':
        algo = KNNBasic(sim_options=sim_options)
    elif algorithm == 'NMF':
        algo = NMF()
    else:
        raise ValueError("Invalid algorithm specified")

    # Train the algorithm
    start_time = time.time()
    algo.fit(trainset)
    elapsed_time = time.time() - start_time

    # Making predictions
    predictions = algo.test(testset)

    # Calculating RMSE and MAE
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Print results
    print(f"RMSE of {algorithm} Algorithm: {rmse}")
    print(f"MAE of {algorithm} Algorithm: {mae}")
    print(f"Time taken to train {algorithm} Algorithm: {elapsed_time} seconds")

    return algo

train_for_context(data_df, 'SVD')
# %%

from surprise import Dataset, Reader, SVD, KNNBasic, NMF, accuracy
from surprise.model_selection import cross_validate
import pandas as pd


def train_for_context_with_cross_validation(dataframe, algorithm, sim_options=None, cv_folds=5):
    reader = Reader(rating_scale=(dataframe['rating'].min(), dataframe['rating'].max()))
    data = Dataset.load_from_df(dataframe[['user', 'item', 'rating']], reader)

    # Initialize the algorithm
    if algorithm == 'SVD':
        algo = SVD()
    elif algorithm == 'KNNBasic':
        algo = KNNBasic(sim_options=sim_options)
    elif algorithm == 'NMF':
        algo = NMF()
    else:
        raise ValueError("Invalid algorithm specified")

    # Perform cross-validation and return results
    cross_validation_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=cv_folds, verbose=True)

    # Calculate and return the average RMSE and MAE
    avg_rmse = pd.Series(cross_validation_results['test_rmse']).mean()
    avg_mae = pd.Series(cross_validation_results['test_mae']).mean()

    return avg_rmse, avg_mae

# Example usage
avg_rmse, avg_mae = train_for_context_with_cross_validation(data_df, 'SVD')
print(f"Average RMSE: {avg_rmse}")
print(f"Average MAE: {avg_mae}")

# %%
