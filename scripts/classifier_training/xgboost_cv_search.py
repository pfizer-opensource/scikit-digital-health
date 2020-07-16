import pandas as pd
import numpy as np
import random
from scipy.stats import randint, uniform, loguniform, nbinom

import xgboost as xgb

from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_hdf('../feature_exploration/features.h5', key='no_preprocessing')

# take top half of features based on PP score
data = data.drop([
    'Skewness',
    'Kurtosis',
    'LinearSlope',
    'SpectralFlatness',
    'Autocorrelation',
    'RangeCountPercentage',
    'ComplexityInvariantDistance',
    'PowerSpectralSum',
    'RatioBeyondRSigma',
    'SignalEntropy',
    'DominantFrequencyValue',
    'JerkMetric',  # add mean cross rate, remove Jerkmetric (correlation with DimensionlessJerk)
    'StdDev'  # add mean, remove StdDev (high correlation with RMS)
], axis=1)

# get the subjects for which LOSO actually makes sense: those with multiple activities (ie more than just walking)
gbc = data.groupby(['Subject', 'Activity'], as_index=False).count()
loso_subjects = [i for i in gbc.Subject.unique() if gbc.loc[gbc.Subject == i].shape[0] > 3]

random.seed(5)  # fix the generation so that its the same every time
random.shuffle(loso_subjects)

training_masks = []
validation_masks = []
testing_masks = []

for i in range(0, len(loso_subjects), 3):
    tr_m = np.ones(data.shape[0], dtype='bool')
    v_m = np.zeros(data.shape[0], dtype='bool')
    
    for j in range(3):
        tr_m &= data.Subject != loso_subjects[i+j]
    for j in range(2):
        v_m |= data.Subject == loso_subjects[i+j]
    te_m = data.Subject == loso_subjects[i+2]
    
    training_masks.append(tr_m)
    validation_masks.append(v_m)
    testing_masks.append(te_m)
    
# estimator setup/parameter distributions
estimator = xgb.XGBRFClassifier()

param_distributions = {
    'n_estimators': randint(7, 100),
    'max_depth': randint(4, 13),
    'learning_rate': loguniform(1e-5, 0.9),
    'gamma': [0, 0, 0, 0.01, 0.05, 0.25, 0.6, 1],
    'importance_type': ['gain', 'weight', 'cover'],
    'tree_method': ['exact', 'approx', 'hist', 'gpu_hist'],
    'reg_alpha': [0.25, 0.75, 1, 1, 1, 1.25, 2, 4, 5],
    'reg_lambda': [0, 0, 0, 0.1, 0.3, 0.6, 1, 2]
}

n_iter = 100
scoring = {'F1': make_scorer(f1_score), 'Accuracy': make_scorer(accuracy_score)}
n_jobs = -1  # use all possible cores
refit = False  # don't want to refit on whole dataset afterwards
verbose = 2
cv = ((training_masks[i], validation_masks[i]) for i in range(len(training_masks)))

clf = RandomizedSearchCV(
    estimator,
    param_distributions,
    cv=cv,
    n_iter=n_iter,
    scoring=scoring,
    n_jobs=n_jobs,
    refit=refit,
    verbose=verbose
)

search = clf.fit(data.iloc[:, 3:], data.Label)

cv_results = pd.DataFrame(data=search.cv_results_)
cv_results.to_csv('xgbrf_cv_results_topfeats.cvs', index=False)
