"""
Functions for various statistical testing methods or feature selection

Lukas Adamowicz
Pfizer
May 20, 2019
"""
from numpy import unique, zeros, sqrt, median, sum as nsum, argsort, std, mean
from numpy.linalg import norm
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score

__all__ = ['db_2class', 'corr_select', 'cohen_d', 'ttest_select', 'mwu_select', 'auc_score']


def db_2class(x, y):
    classes = unique(y)
    ind = y == classes[0]

    db_feat = zeros(x.shape[1])

    for feat_ind in range(x.shape[1]):
        feat = x[:, feat_ind]
        c1 = median(feat[ind])  # median of class 1
        c2 = median(feat[~ind])  # median of class 2

        s1 = sqrt(nsum((feat[ind] - c1) ** 2) / feat[ind].size)
        s2 = sqrt(nsum((feat[~ind] - c2) ** 2) / feat[ind].size)

        m12 = norm(c1 - c2)

        db_feat[feat_ind] = (s1 + s2) / (m12 + 0.000001)  # add small number to prevent infinity

    db_rank = argsort(db_feat)

    return db_feat, db_rank


def corr_select(x, y):
    r = zeros(x.shape[1])

    for i in range(x.shape[1]):
        r[i], _ = pearsonr(x[:, i], y)

    return r


def cohen_d(x1, x2):
    n1 = x1.size
    n2 = x2.size

    s1 = std(x1, ddof=1)
    s2 = std(x2, ddof=1)

    s = sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

    d = (mean(x1) - mean(x2)) / s
    return d


def ttest_select(x, y):
    classes = unique(y)
    ind = y == classes[0]

    p = zeros(x.shape[1])
    d = zeros(x.shape[1])
    for i in range(x.shape[1]):
        _, p[i] = ttest_ind(x[ind, i], x[~ind, i])
        d[i] = cohen_d(x[ind, i], x[~ind, i])

    return p, d


def mwu_select(x, y):
    classes = unique(y)
    ind = y == classes[0]

    p = zeros(x.shape[1])
    d = zeros(x.shape[1])
    for i in range(x.shape[1]):
        _, p[i] = mannwhitneyu(x[ind, i], x[~ind, i], alternative='two-sided')
        d[i] = cohen_d(x[ind, i], x[~ind, i])

    return p, d


def auc_score(x, y):
    a = zeros(x.shape[1])

    for i in range(x.shape[1]):
        a[i] = roc_auc_score(y, x[:, i])

    return a

