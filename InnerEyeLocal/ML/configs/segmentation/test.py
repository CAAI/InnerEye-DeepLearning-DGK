## !

import pandas as pd
from MEDIcaTe.file_folder_ops import load_pickle

fold = 0

split_file = '/homes/kovacs/project_data/hnc-auto-contouring/nnUNet/nnUNet_preprocessed/Task500_HNC01/splits_final.pkl'
split = load_pickle(split_file)

train_ids_list = split[fold]['train']
test_ids_list = []
val_ids_list = split[fold]['val']

# create a dict from pt id to subject id based. 
path_to_data_set = '/homes/kovacs/project_data/hnc-auto-contouring/inner-eye/dataset.csv'
dataset_df = pd.read_csv(path_to_data_set)

dataset_df_for_dict = dataset_df
dataset_df_for_dict = dataset_df_for_dict.drop_duplicates('subject')
pt_ids = dataset_df_for_dict.loc[:, ['filePath']]['filePath'].str[20:-12]
dataset_df_for_dict.insert(3, 'pt_id', pt_ids)

# map split_file ids (nnUNet) to dataset_df ids (inner-eye)
dict_pt2sub_id = dict(zip(list(dataset_df_for_dict['pt_id']), list(dataset_df_for_dict['subject'])))
train_set = [*map(dict_pt2sub_id.get, train_ids_list)]
val_set = [*map(dict_pt2sub_id.get, val_ids_list)]
test_set = test_set = list(range(836, 836+195+1))