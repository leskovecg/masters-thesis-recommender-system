#%% Generate data matrix
import numpy as np
import pandas as pd
import time
import cProfile
import pstats
import matplotlib.pyplot as plt
import itertools
import importlib
import elderly_recsys_tools as erst
from surprise import SVD, KNNBasic, NMF, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise import Dataset, Reader

# Set the threshold for time measurements
TIME_THRESHOLD = 10  # 10 milliseconds
profile = cProfile.Profile()
profile.enable()

#%% Settings & load data
data_path = 'C:/Users/Gasper/OneDrive/FAKS/MAGISTERIJ/letnik_2/SEMESTER_2/magistrska/Code/Data/'
figs_path = 'C:/Users/Gasper/OneDrive/FAKS/MAGISTERIJ/letnik_2/SEMESTER_2/magistrska/Code/Figs/'
tabs_path = 'C:/Users/Gasper/OneDrive/FAKS/MAGISTERIJ/letnik_2/SEMESTER_2/magistrska/Code/Tabs/'


#%%Helper Functions
def load_data(data_path):
    """
    Function to load necessary data from excel files.
    :param data_path: Path to the data folder
    :return: DataFrames of activities, mental health, physical health, and social health
    """
    activities_df = pd.read_excel(data_path + 'Activities.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
    activities_df.rename(columns={nm:'Ac_' + nm for nm in activities_df.columns}, inplace=True)

    mentalHealth_df = pd.read_excel(data_path + 'MentalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
    mentalHealth_df.rename(columns={nm:'Mh_' + nm for nm in mentalHealth_df.columns}, inplace=True)

    physicalHealth_df = pd.read_excel(data_path + 'PhysicalHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
    physicalHealth_df.rename(columns={nm:'Ph_' + nm for nm in physicalHealth_df.columns}, inplace=True)

    socialHealth_df = pd.read_excel(data_path + 'SocialHealth.xlsx', sheet_name='AllAnswers', index_col='S4').replace(-1, np.nan)
    socialHealth_df.rename(columns={nm:'Sh_' + nm for nm in socialHealth_df.columns}, inplace=True)

    #activityContextGen_df = pd.read_excel(data_path + 'ActivityContextGen_v06.xlsx', sheet_name='ActionLst').replace(-1, np.nan)
    #scores_and_wgt_df = pd.read_excel(data_path + 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')
    #scores_and_wgt_2_df = pd.read_excel(data_path + 'ml_data_with_scores_and_wgt_3_values.xlsx', sheet_name='scores_and_wgt_3_v', header=1)
    #wgt_results_annotations_df = pd.read_excel(data_path + 'wgt_results_annotations_3009_4users.xlsx', sheet_name='wgt_results_annotations_3009_4u', header=1).drop(columns=['Unnamed: 0', 'Column1'])
    all_answers_df = activities_df.join(mentalHealth_df).join(physicalHealth_df, rsuffix='_r').join(socialHealth_df, rsuffix='_r')

    return all_answers_df

#  Create sets and dicts
def create_question_groups():
    """
    Create question groups and other dictionaries for the recommender system.
    :return: Dictionaries with question groups and factors
    """
    
    # uIDs = list(np.sort(scores_and_wgt_df.index))
    activity_qs = ['Ac_AB4_1', 'Ac_AB4_2', 'Ac_AB4_3', 'Ac_AB4_4', 'Ac_AB4_5', 'Ac_AB4_8'] 
    #'activity' = [AB4_1, AB4_2, AB4_3, AB4_4, AB4_5, AB4_8]
    phy_health_qs = ['Ph_AB1_7', 'Ph_AB1_11', 'Ph_AB3', 'Ph_AB6_1', 'Ph_AB6_5', 'Ph_AB7_1', 'Ph_AB7_5', 'Ph_AB4_2', 'Ph_AB4_3', 'Ph_AB4_4']
    #'pyhisicalHealth ' = [AB1_11, AB3, AB6_1, AB6_5, AB7_1, AB7_5, AB4_2, AB4_3, AB4_4]
    ment_helath_qs = ['Mh_A75_2', 'Mh_A75_3', 'Mh_A75_4', 'Mh_A75_5', 'Mh_AB1_14', 'Mh_A82_r1', 'Mh_A82_r3']
    # 'mental_health' = [A75_2, A75_3, A75_4, A75_5, AB1_14, A82_r1, A82_r3, A83_r]
    soc_helath_qs = ['Sh_A83_r', 'Sh_sh_AB98_da_ne']
    
    group_qLst = {
        'activity':activity_qs,
        'phy_health':phy_health_qs,
        'ment_helath':ment_helath_qs,
        'soc_helath':soc_helath_qs
    }

    group_factor_dc = {
        'Activities': 'F2', 
        'PhysicalHealth-organskiSistemi': 'F1',
        'PhysicalHealth-nacinZivljenja': 'F1',
        'MentalHealth-osnovno': 'F1',
        'MentalHealth-visje': 'F3'
    }

    return group_qLst, group_factor_dc

def create_dictionaries(activityContextGen_df):
    """
    Create necessary dictionaries from activity context data.
    :param activityContextGen_df: DataFrame containing activity context
    :return: Several dictionaries for mapping actions, questions, and contexts
    """
    # Create dictionaries
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

    return (actID_singleAct_dc, singleAct_actID_dc, qID_qtxt_dc, qID_singleAct_dc, 
            qID_actID_dc, actID_context_dc, actID_props_dc, qID_Group_dc, 
            singleAct_qID_dc, actID_qID_dc)


scores_and_wgt_df = pd.read_excel(data_path + 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')

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




#Set lists of users and actions accoring to selected aspect
def filter_actions_by_aspect(aspect_groups, singleAct_qID_dc, actID_qID_dc, group_qLst):
    """
    Filters actions and action IDs according to the selected aspect groups.
    
    :param aspect_groups: List of selected aspect groups
    :param singleAct_qID_dc: Dictionary mapping actions to question IDs
    :param actID_qID_dc: Dictionary mapping action IDs to question IDs
    :param group_qLst: Dictionary mapping groups to question lists
    :return: Lists of filtered actions and action IDs
    """
    
    #single_act_lst = [g for g in singleAct_qID_dc]
    aspect_groups_lst = ['activity']

    single_act_lst = []
    
    for group in aspect_groups:
        # Get the question list associated with the current aspect group
        question_list = group_qLst[group]
        # Iterate over singleAct_qID_dc to find matches
        for act, qID in singleAct_qID_dc.items():
            # Check if the question ID associated with this action is in the question list
            if qID in question_list:
                single_act_lst.append(act)

    single_actID_lst = []
    for group in aspect_groups:
        question_list = group_qLst[group]
        # Iterate over actID_qID_dc to find matches
        for actID, qID in actID_qID_dc.items():
            if qID in question_list:
                single_actID_lst.append(actID)

    return single_act_lst, single_actID_lst

#%% Helper Functions
def generate_context(activityContextGen_df):
    """
    Generate the context list from the activity context DataFrame.
    """
    activityContextGen_df['kontekst'] = activityContextGen_df.apply(lambda row: erst.get_context(row['C_T1'], row['C_T2'], row['C_T3']), axis=1)
    context_lst = activityContextGen_df['kontekst'].tolist()
    return context_lst[:-6]

def prepare_data_matrix_inputs(uIDs, seq_actID_lst, uIDs_n=100, acts_n=80):
    """
    Prepare input data subsets for generating the data matrix.
    """
    uIDsIn = uIDs[:uIDs_n]
    seq_actID_lstIn = seq_actID_lst[:acts_n]
    return uIDsIn, seq_actID_lstIn

def precompute_data_frames(uIDsIn, actID_lstIn, dictionaries, all_answers_df, group_qLst, aspect_groups_lst, meth_code):
    """
    Precompute necessary data frames for creating the data matrix.
    """
    actID_score_df = erst.get_actID_score_df(uIDsIn, actID_lstIn, dictionaries['actID_qID_dc'], uID_scores_dc, all_answers_df, aspect_groups_lst, meth_code)
    actID_compat_df = erst.get_actIDPair_compat_df(actID_lstIn, dictionaries['qID_Group_dc'], dictionaries['actID_qID_dc'])
    uID_qID_answers_df = erst.get_uID_answers_df(all_answers_df, group_qLst, aspect_groups_lst)
    
    return {
        'actID_score_df': actID_score_df,
        'actID_compat_df': actID_compat_df,
        'uID_qID_answers_df': uID_qID_answers_df
    }

def create_uID_actID_answers_df(uID_qID_answers_df, actID_lstIn, actID_qID_dc):
    """
    Create DataFrame mapping user IDs to action IDs based on their answers.
    """
    uID_actID_answers_df = pd.DataFrame(index=uID_qID_answers_df.index)
    for actID in actID_lstIn:
        if actID_qID_dc[actID] in uID_qID_answers_df:
            uID_actID_answers_df[actID] = uID_qID_answers_df[actID_qID_dc[actID]]
        else:
            print(f'Error: {actID}')
    return uID_actID_answers_df

def evaluate_data_matrix_generation(uIDsIn, seq_actID_lstIn, data_matrix_args, meth_code, r_T=0.3):
    """
    Evaluate data matrix generation time and save the results.
    """
    start_time = time.time()
    # Unpack arguments explicitly and in the correct order required by get_dataMat()
    uID_actID_answers_df, actID_score_df, actID_compat_df = data_matrix_args
    D_lst = erst.get_dataMat(uIDsIn, seq_actID_lstIn, uID_actID_answers_df, actID_score_df, actID_compat_df, r_T, meth_code)
    elapsed_time = time.time() - start_time
    data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating', 'context'])
    print(f"Time taken to get data matrix: {elapsed_time} seconds")

    # Save DataFrame to CSV file
    data_df.to_csv('data_matrix_2.csv', index=False)
    print(f"Data matrix saved to 'data_matrix_2.csv'")


def save_results(D_lst, tabs_path):
    """
    Save results to files.
    """
    dc_lst = [{'Ver': 1, 'P':0.3, 'R':0.6, 'F':0.45}, {'Ver': 1, 'P':0.3, 'R':0.6, 'F':0.45}]
    recSys_succ_df = pd.DataFrame(dc_lst)
    recSys_succ_df.to_excel(tabs_path + 'recSys_succ_df.xlsx')
    recSys_succ_df.to_latex(tabs_path + 'recSys_succ_df.tex')
    print(f"Results saved to {tabs_path}")

def evaluate_models(data_df):
    # Evaluate models with and without context
    print("Evaluating models without context:")
    evaluate_algorithm(data_df, 'SVD')
    evaluate_algorithm(data_df, 'KNNBasic', sim_options={'name': 'cosine', 'user_based': True})
    evaluate_algorithm(data_df, 'NMF')

# Training and Evaluation Functions
def evaluate_algorithm(data, algorithm, sim_options=None):
    """
    Evaluate a specified algorithm using the Surprise library.
    """
    # Splitting the data into training and testing sets (80%-20% split)
    trainset, testset = train_test_split(data, test_size=0.2)

    # Choose the algorithm based on input
    algo = select_algorithm(algorithm, sim_options)
    start_time = time.time()
    algo.fit(trainset)
    elapsed_time = time.time() - start_time

    # Making predictions and calculating RMSE and MAE --> v DF iz dcja 
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    # Print results
    print(f"RMSE of {algorithm} Algorithm: {rmse}")
    print(f"MAE of {algorithm} Algorithm: {mae}")
    print(f"Time taken to train {algorithm} Algorithm: {elapsed_time} seconds")

def select_algorithm(algorithm, sim_options=None):
    """
    Select the appropriate algorithm for training.
    """
    # Choose the algorithm based on input
    if algorithm == 'SVD':
        return SVD()
    elif algorithm == 'KNNBasic':
        if sim_options is None:
            sim_options = {'name': 'cosine', 'user_based': True}
        return KNNBasic(sim_options=sim_options)
    elif algorithm == 'NMF':
        return NMF()
    else:
        raise ValueError("Invalid algorithm specified")



#%% Training Functions for Context-Based Models
def train_for_context(dataframe, algorithm, sim_options=None):
    # Train models for specific contexts
    reader = Reader(rating_scale=(dataframe['rating'].min(), dataframe['rating'].max()))
    data = Dataset.load_from_df(dataframe, reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = select_algorithm(algorithm, sim_options)
    algo.fit(trainset)

    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print(f"RMSE of {algorithm} Algorithm: {rmse}")
    print(f"MAE of {algorithm} Algorithm: {mae}")

    return algo

def train_for_context_with_cross_validation(dataframe, algorithm, sim_options=None, cv_folds=5):
    # Train models using cross-validation for better evaluation
    reader = Reader(rating_scale=(dataframe['rating'].min(), dataframe['rating'].max()))
    data = Dataset.load_from_df(dataframe[['user', 'item', 'rating']], reader)
    algo = select_algorithm(algorithm, sim_options)

    cross_validation_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=cv_folds, verbose=True)
    avg_rmse = pd.Series(cross_validation_results['test_rmse']).mean()
    avg_mae = pd.Series(cross_validation_results['test_mae']).mean()

    print(f"Average RMSE: {avg_rmse}")
    print(f"Average MAE: {avg_mae}")

    return avg_rmse, avg_mae


#%% Example Data Matrix Evaluation


'''
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
'''
# %%
'''
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
'''

# %%
'''
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
'''



#%% Main execution
def main():

    all_answers_df = load_data(data_path)
    group_qLst, group_factor_dc = create_question_groups()

    # Load the activity context data
    activityContextGen_df = pd.read_excel(data_path + 'ActivityContextGen_v06.xlsx', sheet_name='ActionLst').replace(-1, np.nan)

    scores_and_wgt_df = pd.read_excel(data_path + 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')

    dictionaries = create_dictionaries(activityContextGen_df)
    
    (actID_singleAct_dc, singleAct_actID_dc, qID_qtxt_dc, qID_singleAct_dc, qID_actID_dc, actID_context_dc, actID_props_dc, qID_Group_dc, singleAct_qID_dc, actID_qID_dc) = create_dictionaries(activityContextGen_df)

    # Filtering actions and action IDs based on selected aspect groups
    aspect_groups = ['activity']  # Modify as needed
    filtered_single_acts, single_actID_lst = filter_actions_by_aspect(aspect_groups, singleAct_qID_dc, actID_qID_dc, group_qLst)

    # Reloading and further processing
    importlib.reload(erst)
    context_lst = generate_context(activityContextGen_df)

    # Generating sequences of actions
    seq_actID_lst = erst.get_list_of_actions(single_actID_lst, act_max_len = 2)

    # @brief compute data matrix
    # Assumptions:
    # - each action belongs to a question => higher anwser to this questio n
    #   means higher relevance to this activity for this user
    # - each user has his score for each question and higher score to
    #   this question means higher relevance to this group of activities (=question)
    # - relevance r = score * anwser 
    # - if obtained r is positive, we add it to the matrix
    #   

    # Compute data matrix
    # uIDs_n, acts_n = 100, 80
    scores_and_wgt_df = pd.read_excel(data_path + 'ml_data_scores_and_wgt.xlsx', sheet_name='scores_and_wgt', header=1, index_col='person_id')

    uIDs = list(np.sort(scores_and_wgt_df.index))

    uIDsIn, seq_actID_lstIn = prepare_data_matrix_inputs(uIDs, seq_actID_lst)
    actID_lstIn = [act[0] for act in seq_actID_lstIn if len(act) == 1]
    # uIDsIn, seq_actID_lstIn = uIDs[:uIDs_n], seq_actID_lst[:acts_n]
    meth_code = 'score'
    r_T = 0.3 # Threshold

    aspect_groups_lst = ['activity']

    uID_scores_dc['activity']

    # Pass the required dictionaries to precompute_data_frames
    dictionaries = {
        'actID_qID_dc': actID_qID_dc,
        'qID_Group_dc': qID_Group_dc,
        # Add other needed dictionaries as per function requirements
    }

    # Precomputed data frames
    data_frames = precompute_data_frames(uIDsIn, actID_lstIn, dictionaries, all_answers_df, group_qLst, aspect_groups_lst, meth_code)
    uID_actID_answers_df = create_uID_actID_answers_df(data_frames['uID_qID_answers_df'], actID_lstIn, dictionaries['actID_qID_dc'])

    # Generating the data matrix
    D_lst = erst.get_dataMat(uIDsIn, seq_actID_lstIn, uID_actID_answers_df, data_frames['actID_score_df'], data_frames['actID_compat_df'], r_T, meth_code)
    print(f"Generated data matrix with {len(D_lst)} entries.")

    # Call with explicit parameters
    evaluate_data_matrix_generation(
        uIDsIn,
        seq_actID_lstIn,
        [uID_actID_answers_df, data_frames['actID_score_df'], data_frames['actID_compat_df']],
        meth_code=meth_code,
        r_T=r_T
    )

    # Evaluate models
    data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating'])
    evaluate_models(data_df)

    # Save the results
    save_results(D_lst, tabs_path)

if __name__ == "__main__":
    main()








"""
on predlaga:


dc_lst = []
c_dc = {'RMSE':RMSE, 'P':P}
dc_lst.append(c_dc)
eval_df = pd.DataFrane(dc_lst)


"""

