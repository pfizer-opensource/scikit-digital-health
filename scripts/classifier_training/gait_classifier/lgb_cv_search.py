import pandas as pd
import numpy as np
import random
from scipy.stats import randint, dlaplace

import lightgbm as lgb

from sklearn.metrics import f1_score, precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_hdf('../feature_exploration/features.h5', key='incl_stairs')

# get the subjects for which LOSO actually makes sense: those with multiple activities (ie more than just walking)
gbc = data.groupby(['Subject', 'Activity'], as_index=False).count()
loso_subjects = [i for i in gbc.Subject.unique() if gbc.loc[gbc.Subject == i].shape[0] > 3]

random.seed(5)  # fix the generation so that its the same every time
random.shuffle(loso_subjects)

training_masks = []
validation_masks = []
testing_masks = []

for i in range(0, len(loso_subjects), 4):
    tr_m = np.ones(data.shape[0], dtype='bool')
    v_m = np.zeros(data.shape[0], dtype='bool')
    te_m = np.zeros(data.shape[0], dtype='bool')
    
    for j in range(3):
        tr_m &= data.Subject != loso_subjects[i+j]
    for j in range(2):
        v_m |= data.Subject == loso_subjects[i+j]
    for j in range(2):
        te_m |= data.Subject == loso_subjects[i+j+2]
    
    training_masks.append(tr_m)
    validation_masks.append(v_m)
    testing_masks.append(te_m)
    
# CV setup
estimator = lgb.LGBMClassifier(random_state=928)
params = {
    'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],
    'num_leaves': dlaplace(0.2, loc=31),
    'max_depth': [-1, -1, -1, 7, 14, 21],
    'learning_rate': [0.1, 0.1, 0.1, 0.1, 0.01, 0.001, 0.2, 0.3],
    'n_estimators': [10, 20, 40, 75, 100, 125]
}

n_iter = 200
scoring = {'F1': make_scorer(f1_score), 'Precision': make_scorer(precision_score)}
n_jobs = -1
refit = False
verbose = 2
cv = tuple(zip(training_masks, validation_masks))

clf = RandomizedSearchCV(
    estimator,
    params,
    cv=cv,
    n_iter=n_iter,
    scoring=scoring,
    n_jobs=n_jobs,
    refit=refit,
    verbose=verbose
)

search = clf.fit(data.iloc[:, 3:], data.Label)

cv_results = pd.DataFrame(data=search.cv_results_)
cv_results.to_csv('lgb_cv_results_incl_stairs.csv', index=False)
