import pandas as pd
import numpy as np
import random
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_hdf('../feature_exploration/features.h5', key='no_preprocessing')

# take top half of features based on PP score
dtophalf = data.drop([
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

masks = (training_masks, validation_masks, testing_masks)


estimator = RandomForestClassifier()
param_distributions = {
    'n_estimators': randint(5, 100),
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 7, 10, 13, 16, None],
    'min_samples_split': [2, 10, 20, 50, 100, 500, 1000],
    'min_samples_leaf': [1, 2, 4, 8, 16, 32]
}
n_iter = 25
scoring = make_scorer(f1_score)
n_jobs = -1  # use all possible cores
refit = False  # don't want to refit on the whole dataset afterwards
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

search = clf.fit(dtophalf.iloc[:, 3:], dtophalf.Label)

cv_results = pd.DataFrame(data=search.cv_results_)
cv_results.to_csv('rfc_cv_results_topfeats.csv')
