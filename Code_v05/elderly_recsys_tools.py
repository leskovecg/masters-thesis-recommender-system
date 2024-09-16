# File Tools for elderly recommender system

#%% 
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler


#%% Get list of actions
# @brief Get list of actions
def get_list_of_actions(single_act_lst, act_max_len):

    seq_act_lst = []

    for act_len in range(1, act_max_len+1):
        seq_act_lst += list(itertools.combinations(single_act_lst, act_len))

    return seq_act_lst

#single_act_lst, act_max_len = ['a','b','c','d','e'], 3
#x = get_list_of_actions(single_act_lst, act_max_len)


#%% 
def get_context(c_T1, c_T2, c_T3):
    valid_contexts = {'dopoldne', 'popoldne', 'zvečer'}  
    context = []
    
    for c in [c_T1, c_T2, c_T3]:
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

    # Select scores
    Sc_df = pd.DataFrame(index=uIDs, columns=actID_lst)
    for group in aspect_groups_lst:
        c_uID_activity_scores_dc = uID_scores_dc[group]
    
        for uID in uIDs:
            for actID in actID_lst:
                qID = actID_qID_dc[actID]
                c_score = c_uID_activity_scores_dc[uID]
                c_anws = all_answers_df.at[uID, qID]
                if isinstance(c_score, (int, float)) & isinstance(c_anws, (int, float)):
                    Sc_df.at[uID, actID] = c_score * c_anws
                else:
                    Sc_df.at[uID, actID] = 0
                


    # Normalize

    return Sc_df

# @brief: compute dataframe of action pair compatibilities
def get_actIDPair_compat_df(actID_lst, qID_Group_dc, actID_qID_dc):

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



#%%
# @brief estimate rating = score of a sequence of actions act_seq
# Assumptions
# 
# ToDo: normalize contributions to [0,1]
def get_score_estimation(uID, act_seq, uID_actID_answers_df, actID_score_df, compat_df, meth_code):
    
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

    # all_answers_df.replace(r'^([A-Za-z]|[0-9]|_)+$', np.nan, regex=True).astype(float)

    # All qs
    all_qs_lst = []
    for group in aspect_groups_lst:
        all_qs_lst = all_qs_lst + group_qLst[group]

    X_df = pd.DataFrame(index=all_answers_df.index, columns=all_qs_lst)
    X_df.fillna(0, inplace=True)
    all_answers_nums_df = all_answers_df.replace(r' ', np.nan, regex=True).astype(float)

    
    if aspect_groups_lst != []:
        for group in aspect_groups_lst: # For each group

            # Read anwsers
            c_X_df = pd.DataFrame(index=all_answers_df.index)
            for qID in group_qLst[group]:
                c_X_df[qID] = all_answers_nums_df[qID]
            

            # Scale it
            scaler = MinMaxScaler(feature_range=(0, 1))
            c_scl_X_np = scaler.fit_transform(c_X_df.to_numpy())
            c_scl_X_np = pd.DataFrame(data=c_scl_X_np, index=all_answers_df.index, columns=group_qLst[group])
            
            # Add it 
            X_df = X_df.add(c_X_df, fill_value=0)

    # Scale all
    scaler = MinMaxScaler(feature_range=(0, 1))
    scl_X_np = scaler.fit_transform(X_df.to_numpy())
    scl_X_df = pd.DataFrame(data=scl_X_np, index = all_answers_df.index, columns=all_qs_lst)
    scl_X_df.index.names = ['uID']

    return scl_X_df


#uIDsIn, seq_act_lstIn = uIDs[:5], seq_act_lst[:5]
#meth_code = 'score'
#actID_lstIn = list(actID_qID_dc.keys())[:5]
#actID_score_df = erst.get_actID_score_df(uIDsIn, actID_lstIn, actID_qID_dc, uID_scores_dc, all_answers_df, group_str, meth_code)
#print(actID_score_df)
#compat_df = erst.get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)
#print(comp_df)
#uID, act_seq = uIDsIn[0], seq_act_lstIn[0]
#c_rr = get_score_estimation(uID, act_seq, actID_score_df, compat_df, meth_code)

#%% 
# @brief get data matrix, assumptions are given above
# @par meth_code 
#   'score': compatibility and score
def get_dataMat(uIDs, seq_act_lst, uID_actID_answers_df, actID_score_df, compat_df, r_T, meth_code, context_lst = 0):
    

    #D_df = pd.DataFrame(index=uIDs, columns=seq_act_lst, dtype=np.float16)
    D_lst = []

    #if context_lst is not 0:
    #    context_cycle = itertools.cycle(context_lst)

    for uID in uIDs:
        for act_seq in seq_act_lst:

            # Rating = score for this action
            c_r = get_score_estimation(uID, act_seq, uID_actID_answers_df, actID_score_df, compat_df, meth_code)

            if c_r > r_T:
                D_lst.append([uID, act_seq, c_r])
    
    #return D_df
    return D_lst
    


#uIDsIn = uIDs[:10]
#actID_lstIn = list(actID_qID_dc.keys())[:5]
#group_str = 'activity'
#meth_code = 'score' 
#actID_score_df = get_actID_score_df(uIDsIn, actID_lstIn, actID_qID_dc, uID_scores_dc, all_answers_df, group_str, meth_code)
#print(actID_score_df)
#comp_df = get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)
#print(comp_df)

'''
def get_rating_estimation(uID, act_seq, singleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code):
    
    c_r = 0 
    if meth_code == 'score':
        for act in act_seq:   
            c_qID = singleAct_qID_dc[act]
            c_score = uID_activity_scores_dc[uID]
            # c_loading = uID_activity_loading_dc[uID]  # ToDo: include loadings
            c_qa = all_answers_df.at[uID, c_qID]
            if not isinstance(c_qa, (int, float)):
                c_qa = 0
            #print (c_score, c_qa)
            c_r += c_score * c_qa # * c_loading

    return c_r
'''