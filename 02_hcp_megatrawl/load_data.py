"""
HCP MegaTrawl netmats data loader
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import re
from pathlib import Path
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd 

from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder
#from nilearn.input_data import NiftiMapsMasker
#from nilearn.datasets import fetch_atlas_msdl

class config:
    # define some paths
    data_dir = Path(__file__).resolve().parents[0] / 'data'
  

def load_netmats(n_subjects=-1, n_runs=-1, merge=True):
    """ Loads example data from (data/preprocessed)
    """
    logger = logging.getLogger(__name__)
    logger.info('load_netmats(n_subjects={}, n_runs={})'
            .format(n_subjects, n_runs))
    
    # file path (avmovie only, for now)
    # [1] per-subject data 
    #glob_str = "sub-??/in_bold3Tp2/sub-*_task-avmovie_run-*_bold.nii.gz"
    glob_str = "data_hcp_netmats2.txt"
    data_paths = sorted(config.data_dir.glob(glob_str))

    # [2] meta data
    #     - actual timing: "emotions_av_shots_thr50.tsv"
    #     - sampled at 1s: "emotions_av_1s_thr50.tsv"
    #glob_str = "emotions_av_shots_thr50.tsv"
    glob_str = "meta_data_hcp_1003.txt"
    meta_paths = sorted(config.data_dir.glob(glob_str))

    # convert Path objects to strings
    data_paths = [str(_) for _ in data_paths]
    meta_paths = [str(_) for _ in meta_paths]
    
    # check sizes
    logger.debug('found {} data files'.format(len(data_paths)))
    logger.debug('found {} meta data files'.format(len(meta_paths)))
    
    # limit to n_subjects, n_runs
    data_paths = np.r_[data_paths]
    meta_paths = np.r_[meta_paths]



    # load data ?
    logger.info("Loading data...")
    dataset = []



    # load data ?
    logger.info("Loading data...")
    for i, data_path in enumerate(data_paths):

        df_data = pd.read_csv(data_path, sep=" ", header=None)
        df_meta = pd.read_csv(meta_paths[i])
        
        # index
        # df_meta =df_meta.set_index(['subject_id', 'run_id'])

        # encode string variables
        encoders = defaultdict(LabelEncoder)
        for col in df_meta.columns:
            try:
                df_meta[col].astype(float)
            except Exception as e:

                logging.info(" ** Encoding meta column: {}".format(col))
                df_meta[col] = encoders[col].fit_transform(df_meta[col])
    
        # mask data
        #masker = NiftiMasker().fit(mask_paths[0])
        
        # save masker, x
        dataset.append(Bunch(
            data=df_data,
            meta=df_meta,
            encoders=encoders,
            X=df_data.values,
            y=df_meta.values,
            ))
    
    
    # return dataset as Bunch
    if merge:
        dataset = Bunch(
            data=pd.concat((_.data for _ in dataset), sort=False).fillna(0.0),
            meta=pd.concat((_.meta for _ in dataset), sort=False),
            encoders=[_.encoders for _ in dataset],
            )
        dataset.X = dataset.data.values.reshape(dataset.data.shape[0], dataset.data.shape[-1])
        dataset.y = dataset.meta.values.reshape(dataset.meta.shape[0], dataset.meta.shape[-1])

    return dataset




   
