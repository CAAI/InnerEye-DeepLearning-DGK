from pathlib import Path
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import HeadAndNeckBase
from InnerEye.ML.config import PhotometricNormalizationMethod
from MEDIcaTe.file_folder_ops import load_pickle
from InnerEye.ML.utils.split_dataset import DatasetSplits
import pandas as pd


class HNC_tumor_dgk_HeadAndNeckBase_f2(HeadAndNeckBase):
    def __init__(self) -> None:
        super().__init__(
            ground_truth_ids=["tumor"],
            image_channels=["ct", "pet"],
            norm_method=PhotometricNormalizationMethod.Unchanged,
            local_dataset=Path("/homes/kovacs/project_data/hnc-auto-contouring/inner-eye"),
            num_gpus=1,
            num_workers=10,
            num_dataload_workers=10
            )

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        # load dataset split - first path to file
        '''
        Note: To run on the same validation set for fold 0 all models during prediction dataset_df for fold 0 needs to be used.
        During training the fold should be the one you want to train on, i.e. in this case mathing the folk provided in the filename (0,1,2,3,4)
        This note is only relevant when the following predict function is used from commandline:
        python InnerEyeLocal/ML/runner.py --model=HNC_tumor_dgk_HeadAndNeckBase --no-train --local_weights_path=.../last.ckpt --inference_on_val_set=True
        '''
        fold = 0 # during training: set to 2
    
        split_file = '/homes/kovacs/project_data/hnc-auto-contouring/nnUNet/nnUNet_preprocessed/Task500_HNC01/splits_final.pkl'
        split = load_pickle(split_file)
        
        train_ids_list = split[fold]['train']
        test_ids_list = []
        val_ids_list = split[fold]['val']

        #create a dict from pt id to subject id based. 
        dataset_df_for_dict = dataset_df
        dataset_df_for_dict = dataset_df_for_dict.drop_duplicates('subject')
        dataset_df_for_dict['pt_id'] = dataset_df_for_dict['filePath'].str[20:-12]

        # map split_file ids (nnUNet) to dataset_df ids (inner-eye)
        dict_pt2sub_id = dict(zip(list(dataset_df_for_dict['pt_id']), list(dataset_df_for_dict['subject'])))
        train_set = [*map(dict_pt2sub_id.get, train_ids_list)]
        val_set = [*map(dict_pt2sub_id.get, val_ids_list)]

        return DatasetSplits.from_subject_ids(dataset_df, 
                                              train_ids=train_set,
                                              test_ids=test_ids_list,
                                              val_ids=val_set)

# Note of runs:
# /outputs/2022-05-25T134739Z_HNC_tumor_dgk_HeadAndNeckBase run was with norm_method=PhotometricNormalizationMethod.Unchanged on normalized data, num_data_workers=20 and num_gpus=1
# /outputs/2022-05-27T091802Z_HNC_tumor_dgk_HeadAndNeckBase run was with norm_method=PhotometricNormalizationMethod.Unchanged on normalized data, num_data_workers=20 and num_gpus=1
# /outputs/2022-05-28T111738Z_HNC_tumor_dgk_HeadAndNeckBase run was just with its standard norm for both PET and CT. I didn't use normalized data for this run, num_data_workers=64 and it crashed.
# TODO 31-05-2022: Do run from may 28 again but this time do it with num_workers=20 and num_gpus=1.

# TODO 24-05-2022: Normalize all the data as for DeepMedic and train on that. 

# if needed: ssh kovacs@10.49.144.33

# Running the model:
# screen -S/r inner-eye
# export CUDA_VISIBLE_DEVICES=X
# conda activate InnerEye

# cd ~/project_scripts/hnc_segmentation/inner-eye-ms-oktay/InnerEye-DeepLearning
# then run:
# export PYTHONPATH=`pwd`
# python InnerEyeLocal/ML/runner.py --model=HNC_tumor_dgk_HeadAndNeckBase_f2

# https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/building_models.md

# adding number_of_cross_validation_splits=5 gave:
# NotImplementedError: Offline cross validation is only supported for classification models.

'''
Documentation for the restrict_subjects parameter:
"Use at most this number of subjects for train, val, or test set (must be > 0 or None). "
                         "If None, do not modify the train, val, or test sets. If a string of the form 'i,j,k' where "
                         "i, j and k are integers, modify just the corresponding sets (i for train, j for val, k for "
                         "test). If any of i, j or j are missing or are negative, do not modify the corresponding "
                         "set. Thus a value of 20,,5 means limit training set to 20, keep validation set as is, and "
                         "limit test set to 5. If any of i,j,k is '+', discarded members of the other sets are added "
                         "to that set."
restrict_subjects='+,,0', # no testdata since I have already created a separate test-set
'''
