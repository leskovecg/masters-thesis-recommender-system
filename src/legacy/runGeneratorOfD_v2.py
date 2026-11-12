#%% Generate data matrix
import numpy as np
import pandas as pd
import itertools
import scipy.sparse as ss
import pickle
import time
import cProfile
import pstats
import matplotlib.pyplot as plt

# Set the threshold for time measurements
TIME_THRESHOLD = 10  # 10 milliseconds

profile = cProfile.Profile()
profile.enable()

#%% Settings & load data
data_path = 'C:/Users/Gasper/OneDrive/FAKS/MAGISTERIJ/letnik_2/SEMESTER_2/magistrska/koda/Data/'

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

single_act_lst, act_max_len = ['a','b','c','d','e'], 3
x = get_list_of_actions(single_act_lst, act_max_len)

#%%
####################################################################################
import pandas as pd

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

# df = pd.read_excel('C:\\Users\\Gasper\\OneDrive\\FAKS\\MAGISTERIJ\\letnik_2\\SEMESTER_2\\magistrska\\ElderlyActivityContextGen_v01.xlsx')  
# df = pd.read_excel('C:\\Users\\Gasper\\OneDrive\\FAKS\\MAGISTERIJ\\letnik_2\\SEMESTER_2\\magistrska\\ElderlyActivityContextGen_v01.xlsx')  
df = pd.read_excel(data_path + 'ActivityContextGen_v04.xlsx', sheet_name='ActionLst').replace(-1, np.nan)

df['kontekst'] = df.apply(lambda row: get_context(row['C_T1'], row['C_T2'], row['C_T3']), axis=1)

# print(df[['Single_action', 'C_T1', 'C_T2', 'C_T3', 'kontekst']])

kontekst_list = df['kontekst'].tolist()

# Now cut the last 6 elements
kontekst_list = kontekst_list[:-6]

####################################################################################

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
def get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list = 0):
    
    r_T = 0 # rating threshold

    #D_df = pd.DataFrame(index=uIDs, columns=seq_act_lst, dtype=np.float16)
    D_lst = []

    if kontekst_list is not 0:
        context_cycle = itertools.cycle(kontekst_list)

    for uID in uIDs:
        for act_seq in seq_act_lst:

            if kontekst_list is not 0:
                # Access the context using index j
                context = next(context_cycle)  # Get next context

            # Rating for this action
            raw_r = get_rating_estimation(uID, act_seq, asingleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
            # Compatibility score
            comp_score = get_compatibility_score(act_seq)

            if raw_r <= r_T:
                r = np.nan
            else:
                r = comp_score * raw_r
            #D_df.at[uID, act_seq] = r
            if kontekst_list is not 0:
                # Append the tuple including context to the list
                D_lst.append([uID, act_seq, r, context])
            else:
                D_lst.append([uID, act_seq, r])
    
    #return D_df
    return D_lst
    

#uIDsIn, seq_act_lstIn = uIDs[:5], seq_act_lst[:49]
meth_code = 'score'
# D_lst = get_dataMat(uIDs, seq_act_lst, meth_code)
#D_df = get_dataMat(uIDs, seq_act_lst, meth_code)
#D_df.to_excel(data_path + meth_code+'_D_df.xlsx')

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
start_time = time.time()  # Start time
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list)
end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
data_df_context = pd.DataFrame(D_lst, columns=['user', 'item', 'rating','context'])
print(f"Time taken to get data matrix: {elapsed_time} seconds")

start_time = time.time()  # Start time
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list = 0)
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
# %%
from surprise import Dataset
from surprise import Reader

# Assuming D_lst contains tuples of (user, item, rating)
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code)

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
def get_dataMat_with_context(uIDs, seq_act_lst, meth_code, context_df):
    r_T = 0  
    D_lst_with_context = []  

    for i, uID in enumerate(uIDs):
        for j, act_seq in enumerate(seq_act_lst):
            # Get the context for the current user and action sequence
            c = context_list[i]  # Assuming context_list is aligned with uIDs
            # Get the rating for this action
            raw_r = get_rating_estimation(uID, act_seq, asingleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
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
D_lst_with_context = get_dataMat_with_context(uIDs, seq_act_lst, meth_code, kontekst_list)


import pandas as pd
import numpy as np

def get_dataMat_with_context(uIDs, seq_act_lst, meth_code, kontekst_list):
    D_lst_with_context = []  

    for uID in uIDs:
        for j, act_seq in enumerate(seq_act_lst):
            context = kontekst_list[j]  # Access the context using index j
            # Append the required elements to the list
            D_lst_with_context.append([uID, act_seq, meth_code, context])

    # Convert the list to a DataFrame
    df = pd.DataFrame(D_lst_with_context, columns=['user(uIDs)', 'item(seq_act_lst)', 'rating(meth_code)', 'context'])
    return df

# Assuming uIDs, seq_act_lst, meth_code, and kontekst_list are defined
D_lst_with_context_df = get_dataMat_with_context(uIDs, seq_act_lst, meth_code, kontekst_list)
D_lst_with_context_df

# %%

def get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list):
    
    r_T = 0  # Rating threshold
    D_lst = []

    for uID in uIDs:
        for j, act_seq in enumerate(seq_act_lst):
            # Access the context using index j
            context = kontekst_list[j]

            # Rating for this action
            raw_r = get_rating_estimation(uID, act_seq, asingleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
            # Compatibility score
            comp_score = get_compatibility_score(act_seq)

            if raw_r <= r_T:
                r = np.nan
            else:
                r = comp_score * raw_r

            # Append the tuple including context to the list
            D_lst.append([uID, act_seq, r, context])
    
    return D_lst

# Assuming uIDs, seq_act_lst, meth_code, and kontekst_list are defined
D_lst_with_context = get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list)


start_time = time.time()  # Start time
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list)
end_time = time.time()  # End time
elapsed_time = end_time - start_time  # Calculate elapsed time
data_df = pd.DataFrame(D_lst, columns=['user', 'item', 'rating', 'context'])

# Saving DataFrame to CSV file
data_df.to_csv('data_matrix_2.csv', index=False)  # Index set to False to not include row indices in the CSV file


# %%
def get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list = 0):
    
    r_T = 0  # Rating threshold
    D_lst = []

    context_cycle = itertools.cycle(kontekst_list)

    for uID in uIDs:
        for act_seq in seq_act_lst:
            # Access the context using index j
            context = next(context_cycle)  # Get next context

            # Rating for this action
            raw_r = get_rating_estimation(uID, act_seq, asingleAct_qID_dc, uID_activity_scores_dc, all_answers_df, meth_code)
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

# Assuming uIDs, seq_act_lst, meth_code, and kontekst_list are defined
D_lst_with_context = get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list)


start_time = time.time()  # Start time
D_lst = get_dataMat(uIDs, seq_act_lst, meth_code, kontekst_list)
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
