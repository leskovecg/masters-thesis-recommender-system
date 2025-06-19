# File Tools for elderly recommender system
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import KFold, GridSearchCV
import openpyxl
import random
from pathlib import Path

##############################################################
#
# DATA LOADING AND PREPARATION
#
##############################################################
def load_data(D_lst, with_context=False):
    """
    Converts a list of user-item-rating tuples (or 4-tuples with context) into a Surprise dataset.

    Parameters:
    - D_lst (list): Either:
        - A list of 3-tuples/lists in format (user_id, item_id, rating)
        - OR a list of 4-tuples/lists in format (user_id, item_id, context, rating)
    - with_context (bool): Set to True if D_lst contains context and you want to ignore it

    Returns:
    - df (pd.DataFrame): The converted DataFrame version of D_lst
    - data (surprise.Dataset): The Surprise dataset that can be used for training/testing
    """

    if with_context:
        cleaned_lst = [(uid, item[0] if isinstance(item, tuple) else item, rating) 
                       for uid, item, _, rating in D_lst]
    else:
        cleaned_lst = D_lst  # Assume already in correct (user_id, item_id, rating) format

    df = pd.DataFrame(cleaned_lst, columns=['user_id', 'item_id', 'rating'])
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

    return df, data

def load_context_data(filepath):
    """
    Loads contextual information from an Excel sheet and converts it to a dictionary.

    Parameters:
    - filepath (str): Path to the Excel file containing context data

    Returns:
    - context_dc (dict): Dictionary where keys are actIDs and values are context dictionaries
    """

    df = pd.read_excel(filepath, sheet_name='ActionLst').replace(-1, pd.NA)
    context_dc = {}
    for _, row in df.iterrows():
        actID = row['actID']
        context_dc[actID] = {
            'qID': row['qID'],
            'C_T1': row['C_T1'],
            'C_T2': row['C_T2'],
            'C_T3': row['C_T3'],
            'C_P1': row['C_P1'],
            'C_P2': row['C_P2'],
            'C_P3': row['C_P3']
        }

    return context_dc

##############################################################
#
# ACTION SEQUENCE GENERATION
#
##############################################################
def get_list_of_actions(single_act_lst, act_max_len):
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

#single_act_lst, act_max_len = ['a','b','c','d','e'], 3
#x = get_list_of_actions(single_act_lst, act_max_len)

# @brief get data matrix, assumptions are given above
# @par meth_code 
#   'score': compatibility and score
def get_dataMat(uIDs, seq_act_lst, uID_actID_answers_df, actID_score_df, compat_df, r_T, meth_code, context_lst = 0):
    """
    Builds a list of user-action sequence pairs with associated scores above a threshold.

    Parameters:
    - uIDs (list): List of user IDs
    - seq_act_lst (list): List of action sequences
    - uID_actID_answers_df (DataFrame): User answers for action questions
    - actID_score_df (DataFrame): Precomputed action scores per user
    - compat_df (DataFrame): Compatibility scores between actions
    - r_T (float): Threshold to include a recommendation
    - meth_code (str): Method to compute the rating
    - context_lst (optional): Ignored (future use)

    Returns:
    - D_lst (list): List of (user_id, action_sequence, rating) tuples
    """

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

# @brief Get random context 
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
    C_T = random.choice(['C_T1', 'C_T2', 'C_T3'])
    C_P = random.choice(['C_P1', 'C_P2', 'C_P3'])

    c_cntx = {'qID': full_cntx['qID'], 'C_T': full_cntx[C_T], 'C_P': full_cntx[C_P], 'C_A':''}
    
    return c_cntx

# @brief Get random context 
def get_one_random_context(full_cntx):
    """
    Selects a random context (time and place) from a full context definition for a single activity.

    Parameters:
    - full_cntx (dict): Dictionary containing full context fields for one activity,
                        including keys like 'qID', 'C_T1', 'C_T2', 'C_T3', 'C_P1', 'C_P2', 'C_P3'.

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

    C_T = random.choice(['C_T1', 'C_T2', 'C_T3'])
    C_P = random.choice(['C_P1', 'C_P2', 'C_P3'])

    c_cntx = {'qID': full_cntx['qID'], 'C_T': full_cntx[C_T], 'C_P': full_cntx[C_P], 'C_A':''}
    
    return c_cntx

# @brief test if a given action fits to a given context
def is_action_context_feasibleQ(actID, cntx, actID_context_dc):
    """
    Checks whether a given activity (identified by actID) is feasible within a given context.

    This function compares a candidate context (e.g., 'dopoldne', 'doma') to the predefined
    acceptable time and place values for that activity.

    Parameters:
    - actID (str): The ID of the action to be evaluated.
    - cntx (dict): The target context to check, with keys:
                   - 'C_T' (context time), e.g., 'dopoldne'
                   - 'C_P' (context place), e.g., 'kjerkoli'
    - actID_context_dc (dict): A dictionary where each key is an action ID, and the value is
                               a dict with keys like 'C_T1', 'C_T2', 'C_T3', 'C_P1', etc., 
                               defining allowed times and places for that action.

    Returns:
    - bool: True if the context is compatible (both time and place match at least one of the
            allowed ones), otherwise False.
    """
    
    f_cntx = actID_context_dc[actID] # Full context
    f_cntx = actID_context_dc[actID]
    C_Ts = [f_cntx[k].strip() for k in ['C_T1', 'C_T2', 'C_T3'] if isinstance(f_cntx[k], str)]
    C_Ps = [f_cntx[k].strip() for k in ['C_P1', 'C_P2', 'C_P3'] if isinstance(f_cntx[k], str)]

    if (cntx['C_T'] in C_Ts) and (cntx['C_P'] in C_Ps):
        return True 
    else:
        return False
    
##############################################################
#
# RECOMMENDATION LOGIC
#
##############################################################
#@brief returns m best actions triples
def get_recommendations(uID, 
                        n_recommendations=5, 
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
    """

    if D_lst is not None:
        # Method 1: Use precomputed matrix
        user_entries = [x for x in D_lst if x[0] == uID]
        sorted_entries = sorted(user_entries, key=lambda x: x[2], reverse=True)
        return sorted_entries[:n_recommendations]

    elif model is not None and trainset is not None:
        # Method 2: Use model prediction
        all_iids = trainset._raw2inner_id_items.keys()
        predictions = []

        for raw_iid in all_iids:
            act_id = raw_iid[0] if isinstance(raw_iid, tuple) else raw_iid

            # Context filtering
            if context and actID_context_dc:
                if not is_action_context_feasibleQ(act_id, context, actID_context_dc):
                    continue  # Skip actions not matching the context

            est = model.predict(uid=str(uID), iid=act_id).est
            predictions.append((uID, act_id, est))

        sorted_predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
        return sorted_predictions[:n_recommendations]

    else:
        raise ValueError("Provide either D_lst or both model and trainset.")

##############################################################
#
# EVALUATION METRICS AND VALIDATION
#
##############################################################
def evaluate_precision_recall_f1(D_lst, ground_truth_dict, k=5):
    """
    Computes Precision, Recall, and F1-score based on 
    user recommendations and known relevant sequences (ground truth).
    
    :param D_lst: list of dictionaries containing top recommendations per user.
                  Example: [{'user_id': 101, 'top_recommendations': ['seq1', 'seq2', ...]}, ...]
    :param ground_truth_dict: dictionary with relevant sequences for each user.
                  Example: {101: {'seq1', 'seq3'}, 102: {'seq2'}}
    :param k: number of top recommended sequences to consider (default = 5)
    
    :return: average metric values: precision, recall, f1
    """
    precision_list, recall_list, f1_list = [], [], []

    for user in D_lst:
        uid = user['user_id']
        predicted = set(user['top_recommendations'][:k])
        actual = ground_truth_dict.get(uid, set())

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    avg_p = round(sum(precision_list) / len(precision_list), 3)
    avg_r = round(sum(recall_list) / len(recall_list), 3)
    avg_f = round(sum(f1_list) / len(f1_list), 3)

    return avg_p, avg_r, avg_f

def perform_cross_validation(data, algorithm_name='SVD', n_splits=10, test_size=0.25, random_state=42, sim_options=None):
    """
    Performs k-fold cross-validation and evaluates model performance.

    Parameters:
    - data (surprise.Dataset): The data to evaluate
    - algorithm_name (str): Name of the algorithm (e.g., 'SVD')
    - n_splits (int): Number of folds for KFold
    - test_size (float): Size of the test set per fold (not used by Surprise KFold)
    - random_state (int): Seed for reproducibility
    - sim_options (dict): Not used here but can be extended for similarity-based models

    Returns:
    - avg_metrics (dict): A dictionary with average RMSE, MAE, MSE, FCP, and dummy training time
    """

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    algo = initialize_model(algorithm_name)
    metrics = {'rmse': [], 'mae': [], 'fcp': [], 'mse': []}

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        metrics['rmse'].append(accuracy.rmse(predictions, verbose=False))
        metrics['mae'].append(accuracy.mae(predictions, verbose=False))
        metrics['fcp'].append(accuracy.fcp(predictions))
        metrics['mse'].append(accuracy.mse(predictions, verbose=False))

    avg_metrics = {
        'Average RMSE': sum(metrics['rmse']) / n_splits,
        'Average MAE': sum(metrics['mae']) / n_splits,
        'Average FCP': sum(metrics['fcp']) / n_splits,
        'Average MSE': sum(metrics['mse']) / n_splits,
        'Average Training Time': 0  # Can be updated if actual time is measured
    }
    return avg_metrics

def save_evaluation_results(results_list, save_path):
    """
    Saves model evaluation results to an Excel file.

    Parameters:
    - results_list (list): A list of dictionaries containing evaluation results
    - save_path (str): Path to the folder where Excel should be saved (should end with '/')
    """

    df = pd.DataFrame(results_list)

    # Prevod v slovenščino
    df.rename(columns={
        'Algorithm': 'Metoda',
        'Average RMSE': 'Povprečni RMSE',
        'Average MAE': 'Povprečni MAE',
        'Average MSE': 'Povprečni MSE',
        'Average FCP': 'Povprečni FCP',
        'Average Training Time': 'Povprečni čas učenja (s)'
    }, inplace=True)

    # Zaokrožitev na 3 decimalke
    df = df.round(3)

    # Shrani kot Excel za preverjanje (opcijsko)
    df.to_excel(save_path / 'evaluation_results_slo.xlsx', index=False)

    # Shrani kot .tex za uporabo v LaTeX-u
    with open(save_path / 'evaluation_results_slo.tex', 'w', encoding='utf-8') as f:
        f.write(df.to_latex(index=False, float_format="%.3f", escape=False))

##############################################################
#
# MODEL SELECTION AND TRAINING
#
##############################################################
def grid_search(data, algorithm_name, param_grid):
    """
    Performs hyperparameter tuning using grid search with cross-validation.

    Parameters:
    - data (surprise.Dataset): The dataset to train on
    - algorithm_name (str): Name of the algorithm (currently only 'SVD')
    - param_grid (dict): Dictionary of hyperparameters to test

    Returns:
    - gs (GridSearchCV): The trained grid search object containing best parameters
    """

    algo_class = SVD  # Add other algorithms here if needed
    gs = GridSearchCV(algo_class, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    print("Best RMSE score:", gs.best_score['rmse'])
    print("Best params:", gs.best_params['rmse'])
    
    return gs

def initialize_model(algorithm_name='SVD', **kwargs):
    """
    Initializes a Surprise algorithm model based on name and parameters.

    Parameters:
    - algorithm_name (str): The name of the algorithm (currently only supports 'SVD')
    - kwargs: Additional keyword arguments to pass to the algorithm constructor

    Returns:
    - model (surprise.AlgoBase): Initialized Surprise model
    """

    if algorithm_name == 'SVD':
        return SVD(**kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    
##############################################################
#
# EXPORTING RESULTS
#
##############################################################
def save_recommendations_to_excel(recommendations, file_path):
    """
    Saves a list of recommendations to an Excel file.

    Parameters:
    - recommendations (list): List of tuples (user_id, item_id, estimated_rating)
    - file_path (str): Full file path for the Excel file output
    """

    df = pd.DataFrame(recommendations, columns=['user_id', 'item_id', 'estimated_rating'])
    df.to_excel(str(file_path), index=False)  # Pandas zahteva str pot za Excel

#uIDsIn, seq_act_lstIn = uIDs[:5], seq_act_lst[:5]
#meth_code = 'score'
#actID_lstIn = list(actID_qID_dc.keys())[:5]
#actID_score_df = erst.get_actID_score_df(uIDsIn, actID_lstIn, actID_qID_dc, uID_scores_dc, all_answers_df, group_str, meth_code)
#print(actID_score_df)
#compat_df = erst.get_actIDPair_compat_df(actID_lstIn, qID_Group_dc, actID_qID_dc)
#print(comp_df)
#uID, act_seq = uIDsIn[0], seq_act_lstIn[0]
#c_rr = get_score_estimation(uID, act_seq, actID_score_df, compat_df, meth_code)



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

##############################################################
#
# CONTEXT PROCESSING AND TRANSFORMATION
#
##############################################################
def get_context(c_T1, c_T2, c_T3):
    """
    Constructs a context string from three contextual time values.

    Parameters:
    - c_T1, c_T2, c_T3 (str or NaN): Contextual descriptors

    Returns:
    - str: Concatenated context string or 'nd' if none are valid
    """

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

##############################################################
#
# SCORE AND COMPATIBILITY MATRIX GENERATION
#
##############################################################
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
                if isinstance(c_score, (int, float)) & isinstance(c_anws, (int, float)):
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

# @brief estimate rating = score of a sequence of actions act_seq
# Assumptions
# 
# ToDo: normalize contributions to [0,1]
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

##############################################################
#
# USER ANSWER PROCESSING
#
##############################################################
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


