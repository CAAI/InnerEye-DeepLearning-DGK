'''
Purpose: Running training and inference using InnerEye Deep Learning model training on HNC data.

Note: The content of the get_model_train_test_dataset_splits needs to be adapted depending on the training/test data used.

Unfortunately, currently 5 files need to be adapted to do a complete inference with majority voting. These are: 

InnerEye-DeepLearning/InnerEyeLocal/ML/configs/segmentation/HNC_tumor_dgk_HeadAndNeckBase.py
InnerEye-DeepLearning/InnerEyeLocal/ML/configs/segmentation/HNC_tumor_dgk_HeadAndNeckBase_f1.py
InnerEye-DeepLearning/InnerEyeLocal/ML/configs/segmentation/HNC_tumor_dgk_HeadAndNeckBase_f2.py
InnerEye-DeepLearning/InnerEyeLocal/ML/configs/segmentation/HNC_tumor_dgk_HeadAndNeckBase_f3.py
InnerEye-DeepLearning/InnerEyeLocal/ML/configs/segmentation/HNC_tumor_dgk_HeadAndNeckBase_f4.py

I use the model in the following way:
If needed log on to AI machine (ssh kovacs@10.49.144.33 or local machine for me)

Running the model:
# screen -S/r inner-eye -- useful so you can close computer and keep running after
# export CUDA_VISIBLE_DEVICES=X -- select X to match the gpu you want to use. 
# conda activate InnerEye

# cd ~/project_scripts/hnc_segmentation/inner-eye-ms-oktay/InnerEye-DeepLearning (or whereever you located your InnerEye repo)
# then run:
# export PYTHONPATH=`pwd`
# python InnerEyeLocal/ML/runner.py --model=HNC_tumor_dgk_HeadAndNeckBase_fX
'''

from pathlib import Path
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import HeadAndNeckBase
from InnerEye.ML.config import PhotometricNormalizationMethod
from MEDIcaTe.file_folder_ops import load_pickle
from InnerEye.ML.utils.split_dataset import DatasetSplits
import pandas as pd


class HNC_tumor_dgk_HeadAndNeckBase_f4(HeadAndNeckBase):
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
        '''
        Note:

        Inference on validation set:
        To run on the same validation set for fold 0 all models during prediction dataset_df for fold 0 needs to be used.
        During training the fold should be the one you want to train on, i.e. in this case mathing the folk provided in the filename (0,1,2,3,4)
        This note is only relevant when the following predict function is used from commandline:
        python InnerEyeLocal/ML/runner.py --model=HNC_tumor_dgk_HeadAndNeckBase --no-train --local_weights_path=.../last.ckpt --inference_on_val_set=True

        Inference on test set:
        - adding of test and validation data is only to avoid crash of the DatasetSplits.from_sibject_ids function. They are never used.
        - Command to run is in the file /MEDIcaTe/implemented_methods/inner-eye/predict/test/run_test_inference_inner_eye.sh
        - Command line contains the flag --no-train ensuring no training is done
        - Command line contains the flag --inference_on_test_set=True so the inference will be performed on this. 
        - dataset.csv was created before running on test set. Unfortunately it needs to contain test labels to work. 
        '''
        fold = 0  # during training: set to 1
        split_file = '/homes/kovacs/project_data/hnc-auto-contouring/nnUNet/nnUNet_preprocessed/Task500_HNC01/splits_final.pkl'
        split = load_pickle(split_file)

        train_ids_list = split[fold]['train']
        val_ids_list = split[fold]['val']

        # create a dict from pt id to subject id based. 
        dataset_df_for_dict = dataset_df
        dataset_df_for_dict = dataset_df_for_dict.drop_duplicates('subject')
        pt_ids = dataset_df_for_dict.loc[:, ['filePath']]['filePath'].str[20:-12]
        dataset_df_for_dict.insert(3, 'pt_id', pt_ids)

        # map split_file ids (nnUNet) to dataset_df ids (inner-eye)
        dict_pt2sub_id = dict(zip(list(dataset_df_for_dict['pt_id']), list(dataset_df_for_dict['subject'])))
        train_set = [*map(dict_pt2sub_id.get, train_ids_list)]
        val_set = [*map(dict_pt2sub_id.get, val_ids_list)]

        test_set = list(map(str,list(range(836, 836+195+1))))

        datasplit_return = DatasetSplits.from_subject_ids(dataset_df,
                                              train_ids=train_set,
                                              test_ids=test_set,
                                              val_ids=val_set)
                                              
        print(f'datasplit_return = {datasplit_return}')
        return datasplit_return