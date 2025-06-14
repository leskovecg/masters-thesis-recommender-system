#%% Generate data matrix
import numpy as np
import pandas as pd
import itertools
import scipy.sparse as ss


#%% Settings & load data
data_path = './Data/'


activities_df = pd.read_excel(data_path + 'Activities.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
mentalHealth_df = pd.read_excel(data_path + 'MentalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
physicalHealth_df = pd.read_excel(data_path + 'PhysicalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
socialHealth_df = pd.read_excel(data_path + 'SocialHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
all_answers_df = activities_df.join(mentalHealth_df).join(physicalHealth_df, rsuffix='_r').join(socialHealth_df, rsuffix='_r')

activityContextGen_df = pd.read_excel(data_path + 'ActivityContextGen_v04.xlsx', sheet_name='ActionLst').replace(-1, np.nan)

scores_and_wgt_df = pd.read_excel(data_path + 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')
#scores_and_wgt_2_df = pd.read_excel(data_path + 'ml_data_with_scores_and_wgt_3_values.xlsx', sheet_name='scores_and_wgt_3_v', header=1)
#wgt_results_annotations_df = pd.read_excel(data_path + 'wgt_results_annotations_3009_4users.xlsx', sheet_name='wgt_results_annotations_3009_4u', header=1).drop(columns=['Unnamed: 0', 'Column1'])


# %% Create sets and dicts
uIDs = list(np.sort(scores_and_wgt_df.index))
activity_sq = ['AB4_1', 'AB4_2', 'AB4_3', 'AB4_4', 'AB4_5', 'AB4_8']
phy_health_qs = ['AB1_7', 'AB1_11', 'AB3', 'AB6_1', 'AB6_5', 'AB7_1', 'AB7_5', 'AB4_2', 'AB4_3', 'AB4_4']
ment_helath_qs = ['A75_2', 'A75_3', 'A75_4', 'A75_5', 'AB1_14', 'A82_r1', 'A82_r3']
soc_helath_qs = ['A83_r', 'sh_AB98_da_ne']
group_qLst = {'activity':activity_sq,
              'phy_health':phy_health_qs,
              'ment_helath':ment_helath_qs,
              'soc_helath':soc_helath_qs}


group_factor_dc = {'Activities': 'F2', 
                   'PhysicalHealth-organskiSistemi': 'F1',
                   'PhysicalHealth-nacinZivljenja': 'F1',
                   'MentalHealth-osnovno': 'F1',
                   'MentalHealth-visje': 'F3'}


# Dictionaries
curr_qID = np.nan
curr_qIDs = []
qID_qtxt_dc = {}
qID_singleAct_dc = {}
qID_Group_dc = {}
asingleAct_qID_dc = {}
for ind, row in activityContextGen_df.iterrows():
    # New qID
    if row['Code'] not in ['A82_r1', 'A82_r3', 'A83_r']: # Not covered Qs
        if pd.notnull(row['Code']):
            if row['Code'] not in curr_qIDs:
                curr_qID = row['Code']
                curr_qIDs.append(curr_qID)
                qID_qtxt_dc[curr_qID] = row['Question']
                qID_Group_dc[curr_qID] = row['Group']
                qID_singleAct_dc[curr_qID] = []

        # Add vals
        single_act = row['Single_action']
        qID_singleAct_dc[curr_qID].append(single_act)
        asingleAct_qID_dc[curr_qID+'_'+str(single_act)] = curr_qID


single_act_lst = [g for g in asingleAct_qID_dc]

# uIDs to socres
uID_activity_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['a_F2']))
uID_menHealOsn_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_osnovno_F1']))
uID_menHealVisje2_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F2']))
uID_menHealVisje4_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['mh_visje_F4']))
uID_phyHealNacin_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_nacinZivljenja_F1']))
uID_phyHealOrganski_scores_dc = dict(zip(scores_and_wgt_df.index, scores_and_wgt_df['ph_organskiSistemi_F1']))



#%% Get list of actions
# @brief Get list of actions
def get_list_of_actions(single_act_lst, act_max_len):

    seq_act_lst = []
    for act_len in range(1, act_max_len+1):
        seq_act_lst += list(itertools.combinations(single_act_lst, act_len))

    return seq_act_lst

#single_act_lst, act_max_len = ['a','b','c','d','e'], 3
#x = get_list_of_actions(single_act_lst, act_max_len)
#print (len(x))



#%% Test it
act_max_len = 1
seq_act_lst = get_list_of_actions(single_act_lst, act_max_len)

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

# @brief Estimate list of action compatibility score
def get_compatibility_score(act_seq):
    return 1

# @brief estimate rating of a sequence of actions
# ToDo: normalize contributions to [0,1]
def get_rating_estimation(uID, act_seq, asingleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code):
    
    c_r = 0 
    if meth_code == 'score':
        for act in act_seq:   
            c_qID = asingleAct_qID_dc[act]
            c_score = uID_activity_scores_dc[uID]
            # c_loading = uID_activity_loading_dc[uID]  # ToDo: include loadings
            c_qa = all_answers_df.at[uID, c_qID]
            if not isinstance(c_qa, (int, float)):
                c_qa = 0
            #print (c_score, c_qa)
            c_r += c_score * c_qa # * c_loading

    return c_r

# @brief get data matrix, assumptions are given above
# @par meth_code 
#   'score': compatibility and score
def get_dataMat(uIDs, seq_act_lst, meth_code):
    
    r_T = 0 # rating threshold

    #D_df = pd.DataFrame(index=uIDs, columns=seq_act_lst, dtype=np.float16)
    D_lst = []

    for uID in uIDs:
        for act_seq in seq_act_lst:
            # Rating for this action
            raw_r = get_rating_estimation(uID, act_seq, asingleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
            # Compatibility score
            comp_score = get_compatibility_score(act_seq)

            if raw_r <= r_T:
                r = np.nan
            else:
                r = comp_score * raw_r
            #D_df.at[uID, act_seq] = r
            D_lst.append([uID, act_seq, r])
    
    #return D_df
    return D_lst

#uIDsIn, seq_act_lstIn = uIDs[:5], seq_act_lst[:49]
meth_code = 'score'
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code)
#D_df = get_dataMat(uIDs, seq_act_lst, meth_code)
#D_df.to_excel(data_path + meth_code+'_D_df.xlsx')

# Stats
# D \in [696, 232]
# D elts = 161472
# Non nans = 80736, that is 50%
# %%
'''
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
X_sm =  ss.coo_array((data, (row, col)), dtype=np.int8) # shape=(4, 4))
'''
