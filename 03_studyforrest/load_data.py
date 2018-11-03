"""
Studyforrest data loader
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
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import load_img
from nilearn.signal import clean
#from nilearn.datasets import fetch_atlas_msdl

class config:
    # define some paths
    data_dir = Path(__file__).resolve().parents[0] / 'data'

  

def align_meta_to_data(meta, data):
    """ Align studyforrest meta to data. 

    """
    # copy of metadata
    events = meta.assign(
        start=meta.start.astype(int),
        stop=meta.stop.astype(int)
        ).set_index(['start', 'stop'])
    tags = events.pop('tags')    

    # dataframe for entire run, sampled at 1s
    frames = pd.DataFrame(
        index=np.arange(len(data)*2.), 
        columns=[_ for _ in meta.columns if _ not in ['char', 'tags']],
        )
    # get unique characters, one hot encode
    chars = sorted(set(meta.char), key=list(meta.char).index)
    frames = frames.assign(**{_:0 for _ in chars})

    # loop over events
    for (start, stop) in events.index:
        # repeat meta row every 1s, from start to stop
        TRs = np.arange(start, stop)
        event = events.loc[(start, stop)]
        chars = list(event.pop('char'))
        event_mean = event.values.astype(float).mean(axis=0)

        # average any existing data for TRs within event
        frame = frames.loc[TRs, event.columns]
        if frame.notnull().any().any():
            event_mean = frame.add(event_mean).divide(2.0)

        # update
        frames.loc[TRs, ['start','stop']] = [start, stop]
        frames.loc[TRs, event.columns] = event_mean
        frames.loc[TRs, chars] = 1

        # some attrs from data
        #frames.loc[start:stop, 'run_id'] = data.loc[start:stop, 'run_id']
 
    # resampel every 2s
    frames = frames.assign(TR = frames.index.astype(int))
    frames = frames[frames.TR % 2==0]
    frames = frames.iloc[:len(data), :]
    frames = frames.fillna(-2)
    frames = frames.set_index("TR")
    return frames



def load_studyforrest(n_subjects=-1, n_runs=-1, merge=True):
    """ Loads example data from (data/)
    """
    logger = logging.getLogger(__name__)
    logger.info('load_studyforrest(n_subjects={}, n_runs={})'
            .format(n_subjects, n_runs))
    
    # file path (avmovie only, for now)
    # [1] per-subject data 
    #glob_str = "sub-??/in_bold3Tp2/sub-*_task-avmovie_run-*_bold.nii.gz"
    glob_str = "sub*/*_task-avmovie_run-*_bold.csv"
    data_paths = sorted(config.data_dir.glob(glob_str))

    # [2] meta data
    #     - actual timing: "emotions_av_shots_thr50.tsv"
    #     - sampled at 1s: "emotions_av_1s_thr50.tsv"
    #glob_str = "emotions_av_shots_thr50.tsv"
    glob_str = "emotions_av_1s_thr50.tsv"
    meta_paths = sorted(config.data_dir.glob(glob_str))

    # [3] mask paths
    glob_str = "fconn_atlas_150_2mm.nii.gz"
    atlas_paths = sorted(config.data_dir.glob(glob_str))

    # convert Path objects to strings
    data_paths = [str(_) for _ in data_paths]
    meta_paths = [str(_) for _ in meta_paths]
    atlas_paths = [str(_) for _ in atlas_paths]

    # subject, run meta
    sub_ids = re.findall(r"sub-([0-9]+)", str(data_paths))
    run_ids = re.findall(r"run-([0-9]+)", str(data_paths))

    # check sizes
    logger.debug('found {} data files'.format(len(data_paths)))
    logger.debug('      {} subjects'.format(len(set(sub_ids))))
    logger.debug('      {} runs'.format(len(set(run_ids))))
    logger.debug('found {} meta data files'.format(len(meta_paths)))
    logger.debug('found {} mask data files'.format(len(atlas_paths)))

    # limit to n_subjects, n_runs
    data_paths = np.r_[data_paths]
    meta_paths = np.r_[meta_paths]
    atlas_paths = np.r_[atlas_paths]

    sub_ids = np.r_[sub_ids].astype(int)
    run_ids = np.r_[run_ids].astype(int)    
    
    # check valid n_subject,n_runs
    if n_subjects < 1:
        n_subjects = len(set(sub_ids))
    if n_runs < 1:
        n_runs = len(set(run_ids))


    # load data ?
    logger.info("Loading data...")
    dataset = []
    for i, sub_id in enumerate(set(sub_ids)):
        if len(dataset) >= n_subjects:
            break
        
        # mask all runs for subject
        mask = (sub_ids == sub_id) & (run_ids <= n_runs)
        runs = list(zip(run_ids[mask], data_paths[mask]))
        df_data = pd.concat((
            pd.read_csv(_).assign(subject_id=sub_id, run_id=run_id) 
            for run_id,_ in runs
            ), sort=False, ignore_index=True)

        # load meta: convert each value to a dict
        # 192.0    204.0    'char=FORREST ... val_neg=0.00'
        aligned_meta_path = os.path.join(os.path.dirname(data_paths[mask][0]), 'emotions_aligned.csv')
        if not os.path.exists(aligned_meta_path):
            # process...
            to_dict = lambda _: dict(re.findall(r"(\S+)=(\S+) ?", _))
            df_meta = pd.read_table(meta_paths[0], header=None, names=['start','stop',2]) 
    
            # convert string key value pairs to dicts
            df_dict = pd.DataFrame(df_meta.pop(2).map(to_dict).tolist())
            df_meta = df_meta.join(df_dict)
            df_meta = align_meta_to_data(df_meta, df_data)
            print('Saving...', end=' ')
            df_meta.to_csv(aligned_meta_path)
            print(aligned_meta_path)
        else:
            df_meta = pd.read_csv(aligned_meta_path, index_col="TR")
        
        # re index to data
        print(df_meta.shape, df_data.shape)
        df_meta = df_meta.iloc[list(df_data.index)]
        df_meta = df_meta.reset_index(drop=False)
        df_meta = df_meta.assign(
            subject_id=df_data.subject_id.values,
            run_id=df_data.run_id.values
            )
        df_data = df_data.drop(columns=['subject_id', 'run_id'])

        # index
        # df_meta =df_meta.set_index(['subject_id', 'run_id'])

        # encode string variables
        encoders = defaultdict(LabelEncoder)
        for col in df_meta.columns:
            try:
                df_meta[col].astype(float)
            except:
                logging.info(" ** Encoding meta column: ", col)
                df_meta[col] = encoders[col].fit_transform(df_meta[col])
    
        # mask data
        masker = NiftiLabelsMasker(
            labels_img=atlas_paths[0],
            memory="nilearn_cache",
            #low_pass=0.09, high_pass=0.008, 
            t_r=2.0,
            )
        masker = masker.fit()
        
        # low pas filter
        cleaned_ = clean(df_data.values,
            low_pass=0.09, high_pass=0.008, t_r=2.0,
            )
        df_data.iloc[:, :] = cleaned_
        
        # reindex data to match mask
        atlas_max = load_img(atlas_paths[0]).get_data().max()
        atlas_index = np.arange(1, atlas_max + 1).astype(int)
        #df_zeros = pd.DataFrame(df_data * 0, columns=atlas_index)
        #df_zeros[list(df_data.columns.astype(int))] += df_data.values
        #df_data = df_zeros
        df_data = df_data.reindex(columns=atlas_index.astype(str))



        # save masker, x
        dataset.append(Bunch(
            data=df_data.copy().fillna(0.0),
            meta=df_meta.copy(),#.fillna(-2.0),
            encoders=encoders,
            masker=masker,
            atlas=atlas_paths[0],
            X=df_data.values.copy(),
            y=df_meta.values.copy(),
            ))
    
    
    # return dataset as Bunch
    if merge:
        dataset = Bunch(
            data=pd.concat((_.data for _ in dataset), ignore_index=True, sort=False).fillna(0.0),
            meta=pd.concat((_.meta for _ in dataset), ignore_index=True, sort=False),#.fillna(-2.0),
            encoders=[_.encoders for _ in dataset],
            masker=[_.masker for _ in dataset][0],
            atlas=[_.atlas for _ in dataset][0]
            )
        dataset.X = dataset.data.values.reshape(-1, dataset.data.shape[-1])
        dataset.y = dataset.meta.values.reshape(-1, dataset.meta.shape[-1])

    return dataset



   
