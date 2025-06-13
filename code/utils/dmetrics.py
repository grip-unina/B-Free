# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# https://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    t = np.argmin(np.abs(1.-tpr-fpr))
    return (fpr[t] + 1 - tpr[t])/2

def calculate_eer2(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    import numpy as np
    from scipy.interpolate import interp1d
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    err = 1.-tpr-fpr
    val = (fpr + 1 - tpr)/2
    eer = np.float64(interp1d(err, val)(0))
    return eer

def pd_at_far(y_true, y_score, fpr_th):
    '''
    Returns the Pd at fixed FAR for a binary classifier output.
    '''
    import numpy as np
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return np.interp(fpr_th, fpr, tpr)


def balanced_cllr(y_true, y_score):
    '''
    Returns the balanced CLLR for a binary classifier output.
    y_score has to be a logit in natural base
    '''
    import numpy as np
    _, y_true = np.unique(y_true, return_inverse=True)
    y_true = 2*y_true -1

    y_score = np.logaddexp(-y_true * y_score, 0)
    y_score0 = np.mean(y_score[y_true<0])
    y_score1 = np.mean(y_score[y_true>0])
    cllr = (y_score0+y_score1) / (2*np.log(2))
    return cllr

def balanced_nll_binary(y_true, y_score):
    '''
    Returns the balanced NLL for a binary classifier output.
    y_score has to be a logit in natural base
    '''
    import numpy as np
    return balanced_cllr(y_true, y_score) * np.log(2)


def balanced_ece_binary(y_true, y_score, bins=15):
    '''
    Returns the balanced ECE for a binary classifier output.
    y_score has to be a logit in natural base
    '''
    from scipy.special import expit, softmax
    import numpy as np
    
    _, yp, yc = np.unique(y_true, return_inverse=True, return_counts=True)
    sample_weight = 1/yc
    sample_weight = sample_weight[yp]
    assert np.max(yp)==1
    total_weight = sample_weight.sum()

    prob = expit(y_score)
    correctness = (yp==1)
    
    interval = np.floor(bins*prob)
    ece = 0.0
    for _ in range(bins):
        in_bin = (interval==_)
        if np.any(in_bin):
            weight = sample_weight[in_bin]
            accuracy_in_bin = (weight*correctness[in_bin]).mean() / weight.mean()
            avg_confidence_in_bin = (weight*prob[in_bin]).mean() / weight.mean()
            prop_in_bin = weight.sum() / total_weight

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece