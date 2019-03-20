"""
Haxby data loader
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os 

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nilearn.datasets import fetch_haxby
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn.image import load_img
from nilearn.signal import clean


import matplotlib as mpl
import matplotlib.pyplot as plt


###############################################################################        
### some global meta variables
###############################################################################        
TARGET_NAMES = [
	'rest', 
	'scissors', 
	'face', 
	'cat', 
	'shoe', 
	'house', 
	'scrambledpix',
	'bottle', 
	'chair'
	]

###############################################################################        
### loading (per-subject)
###############################################################################        
def load_subject_data(dataset, index=0, mask='mask_vt', sample_mask=None, smoothing_fwhm=4, **kwargs):
    """ Load functional data for a single haxby subject. """
    # extract relevant files
    func_fn = dataset.func[index]
    mask_fn = dataset.get(mask)
    if not isinstance(mask_fn, str):
        mask_fn = mask_fn[index]

    # extract data from func using mask_vt
    masker = NiftiMasker(
        mask_img=mask_fn, sample_mask=sample_mask,
        standardize=True, detrend=True, smoothing_fwhm=smoothing_fwhm,
        low_pass=0.09, high_pass=0.008, t_r=2.5,
        memory="nilearn_cache",
        )
    X = masker.fit_transform(func_fn)
    data = pd.DataFrame(X)    
    
    # return as bunch
    subject = Bunch()
    subject.data = data
    subject.X = X
    subject.masker = masker
    subject.mask = mask_fn
    subject.func = func_fn
    subject.subject_code = os.path.basename(os.path.dirname(func_fn))
    return subject
    

    
def load_subject_meta(dataset, index=0, sessions=None, targets=None, **kwargs):
    """ Load behavioral data for a single haxby subject. """
    
    # load target information as string and give a numerical identifier to each
    meta = pd.read_csv(dataset.session_target[index], sep=" ")
    
    # condition mask
    sessions = sessions or list(set(meta.chunks)) 
    targets = targets or list(set(meta.labels))

    # apply conditions mask
    session_mask = meta.chunks.isin(sessions)
    target_mask = meta.labels.isin(targets)
    condition_mask = (session_mask & target_mask)

    # mask, extract, factorize
    target, session = meta.labels, meta.chunks
    #target, target_names = pd.factorize(target)
    target_names = np.ravel(TARGET_NAMES)
    target = np.stack(map(TARGET_NAMES.index, target))
    meta = meta.assign(session=session, target=target)

    # convert y to one-hot encoding
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    y = encoder.fit_transform(np.c_[target])
    target = pd.DataFrame(y, columns=target_names)

    # define colors, labels
    target_colors = np.arange(len(target.columns))

    # define colormap, norm
    cmap = kwargs.get('cmap', 'nipy_spectral_r')
    cmap = cmap if callable(cmap) else plt.get_cmap(cmap)
    norm =  mpl.colors.Normalize(target_colors.min(), target_colors.max()+1)

    # map colors, labels
    cmap = mpl.colors.ListedColormap([cmap(norm(_)) for _ in target_colors])
    target_colors_hex = [mpl.colors.to_hex(_) for _ in cmap.colors]


    # for naming
    session_code = "sess" + '_'.join(str(_) for _ in sessions)

    # return as bunch
    subject = Bunch()
    subject.meta = meta.loc[condition_mask]
    subject.target = target.loc[condition_mask]
    subject.y = subject.meta.target
    subject.groups = subject.meta.session
    subject.condition_mask = condition_mask
    subject.target_names = list(target_names)
    subject.target_colors = list(target_colors_hex)
    subject.cmap = cmap
    subject.norm = norm
    subject.session_code = session_code
    return subject



def load_subject(dataset, index=0, **kwargs):
    """ Load a single subject from nilearn dataset. """
    
    # load data, meta
    data = load_subject_data(dataset, index=index, **kwargs)
    meta = load_subject_meta(dataset, index=index, **kwargs)

    # combine, return
    data.update(**dict(meta))

    # mask data
    data.data = data.data.loc[meta.condition_mask]
    data.X = data.data.values.copy()
    #data.meta = data.meta.loc[meta.condition_mask]
    #data.target = data.target.loc[meta.condition_mask]

    # rename
    data.name = data.subject_code + '_' + data.session_code
    return data



def iter_subjects(dataset, subjects=-1, **kwargs):
    """ Load subjects in an nilearn dataset. """
    
    # check subjects, default to all
    if subjects is -1:
        subjects = np.arange(len(dataset.func)) + 1
    
    # convert to subject index
    subjects_ = np.ravel(subjects) - 1
        
    # loop over subjects
    for index_ in subjects_:
        subject_ = load_subject(dataset, index=index_, **kwargs)
        yield subject_
    return
        

        
###############################################################################        
### main loading (muliple subjects)
###############################################################################       
def load_haxby(subjects=-1, verbose=0, **kwargs):
    """ Load subjects in an nilearn dataset. """

    # check subjects
    if subjects is -1:
        subjects = [1, 2, 3, 4, 5, 6]

    # fetch data files
    haxby = fetch_haxby(subjects=subjects, fetch_stimuli=True)
    
    # encode description
    haxby.description = bytes(haxby.description).decode('utf-8')
    if verbose > 2:
        print(haxby.description)
    
    # print some info
    if verbose > 0:
        print('Mask nifti image (3D) is located at:', haxby.mask)
        print('Functional nifti image (4D) is located at:', haxby.func[0])
        #print('[dataset]\n  {}'.format('\n  '.join(list(haxby.keys()))))
        
    # store common meta in haxby
    haxby.meta = load_subject_meta(haxby, **kwargs)
    haxby.update(**dict(haxby.meta))

    # assign subjects as a list 
    haxby.subjects = list(iter_subjects(haxby, subjects=-1, **kwargs))

    # print some info
    if verbose > 1:
        for _, data in enumerate(haxby.subjects):
            print("\n[subject: {}]".format(_))
            print("  data has shape: {}".format(data.data.shape))
            print("  target has shape: {}".format(data.target.shape))
            #if verbose > 2:
            #    print("  target (conditions):\n    {}".format('\n    '.join(data.target.columns)))
    
    # return
    return haxby



   
