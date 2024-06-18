"""
Core functionality for feature computation

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
import json
from warnings import warn

from pandas import DataFrame
from numpy import float64, asarray, zeros, sum, moveaxis


__all__ = ["Bank"]


class ArrayConversionError(Exception):
    pass


def get_n_feats(size, index):
    if isinstance(index, int):
        return 1
    elif isinstance(index, (Iterator, Sequence)):
        return len(index)
    elif isinstance(index, slice):
        return len(range(*index.indices(size)))
    elif isinstance(index, type(Ellipsis)):
        return size


def partial_index_check(index):
    if index is None:
        index = ...

    if not isinstance(index, (int, Iterator, Sequence, type(...), slice)):
        raise IndexError(f"Index type ({type(index)}) not understood.")
    if isinstance(index, str):
        raise IndexError("Index type (str) not understood.")

    return index


def normalize_indices(nfeat, index):
    if index is None:
        return [...] * nfeat
    elif not isinstance(index, (Iterator, Sequence)):  # slice, single integer, etc
        return [partial_index_check(index)] * nfeat
    elif all([isinstance(i, int) for i in index]):  # iterable of ints
        return [index] * nfeat
    elif isinstance(index, Sequence):  # able to be indexed
        return [partial_index_check(i) for i in index]
    else:  # pragma: no cover
        return IndexError(f"Index type ({type(index)}) not understood.")


def normalize_axes(ndim, axis, ind_axis):
    """
    Normalize input axes to be positive/correct for how the swapping has to work
    """
    if axis == ind_axis:
        raise ValueError("axis and index_axis cannot be the same")

    if ndim == 1:
        return 0, None
    elif ndim >= 2:
        """
        |  shape | ax | ia |  move1 | ax | ia |  res  | ax | ia | res move |
        |--------|----|----|--------|----|----|-------|----|----|----------|
        | (a, b) |  0 |  1 | (b, a) |  0 |  0 | (bf,) |    |    |          |
        | (a, b) |  0 |  N | (b, a) |  0 |  N | (f, b)|    |    |          |
        | (a, b) |  1 |  0 |        |    |    | (3a,) |    |    |          |
        | (a, b) |  1 |  N |        |    |    | (f, a)|    |    |          |

        |  shape   | ax| ia   |   move1  | ax| ia|   move2  |   res    |    | ia| res move |
        |----------|---|------|----------|---|---|----------|----------|----|---|----------|
        | (a, b, c)| 0 | 1(0) | (b, c, a)|   |   |          | (bf, c)  |  0 | 0 |          |
        | (a, b, c)| 0 | 2(1) | (b, c, a)|   | 1 | (c, b, a)| (cf, b)  |  0 | 1 | (b, cf)  |
        | (a, b, c)| 0 | N    | (b, c, a)|   |   |          | (f, b, c)|    |   |          |
        | (a, b, c)| 1 | 0    | (a, c, b)|   |   |          | (af, c)  |  0 | 0 |          |
        | (a, b, c)| 1 | 2(1) | (a, c, b)|   | 1 | (c, a, b)| (cf, a)  |  0 | 1 | (a, cf)  |
        | (a, b, c)| 1 | N    | (a, c, b)|   |   |          | (f, a, c)|    |   |          |
        | (a, b, c)| 2 | 0    | (a, b, c)|   |   |          | (af, b)  |  0 | 0 |          |
        | (a, b, c)| 2 | 1    | (a, b, c)|   | 1 | (b, a, c)| (bf, a)  |  0 | 1 | (a, bf)  |
        | (a, b, c)| 2 | N    | (a, b, c)|   |   |          | (f, a, b)|    |   |          |

        |  shape     | ax| ia   |   move1     | ia|   move2     |   res       |   | ia| res move  |
        |------------|---|------|-------------|---|-------------|-------------|---|---|-----------|
        |(a, b, c, d)| 0 | 1(0) | (b, c, d, a)|   |             | (bf, c, d)  | 0 | 0 |           |
        |(a, b, c, d)| 0 | 2(1) | (b, c, d, a)| 1 | (c, b, d, a)| (cf, b, d)  | 0 | 1 | (b, cf, d)|
        |(a, b, c, d)| 0 | 3(2) | (b, c, d, a)| 2 | (d, b, c, a)| (df, b, c)  | 0 | 2 | (d, c, df)|
        |(a, b, c, d)| 0 | N    | (b, c, d, a)|   |             | (f, b, c, d)|   |   |           |
        |(a, b, c, d)| 1 | 0    | (a, c, d, b)|   |             | (af, c, d)  |   |   |           |
        |(a, b, c, d)| 1 | 2(1) | (a, c, d, b)| 1 | (c, a, d, b)| (cf, a, d)  | 0 | 1 | (a, cf, d)|
        |(a, b, c, d)| 1 | 3(2) | (a, c, d, b)| 2 | (d, a, c, b)| (df, a, c)  | 0 | 2 | (a, c, df)|
        |(a, b, c, d)| 1 | N    | (a, c, d, b)|   |             | (f, a, c, d)|   |   |           |
        |(a, b, c, d)| 2 | 0    | (a, b, d, c)|   |             | (af, b, d)  |   |   |           |
        |(a, b, c, d)| 2 | 1    | (a, b, d, c)| 1 | (b, a, d, c)| (bf, a, d)  | 0 | 1 | (a, bf, d)|
        |(a, b, c, d)| 2 | 3(2) | (a, b, d, c)| 2 | (d, a, b, c)| (df, a, b)  | 0 | 2 | (a, b, df)|
        |(a, b, c, d)| 2 | N    | (a, b, d, c)|   |             | (f, a, b, d)|   |   |           |
        |(a, b, c, d)| 3 | 0    | (a, b, c, d)|   |             | (af, b, c)  |   |   |           |
        |(a, b, c, d)| 3 | 1    | (a, b, c, d)| 1 | (b, a, c, d)| (bf, a, c)  | 0 | 1 | (a, bf, c)|
        |(a, b, c, d)| 3 | 2    | (a, b, c, d)| 2 | (c, a, b, d)| (cf, a, b)  | 0 | 2 | (a, b, cf)|
        |(a, b, c, d)| 3 | N    | (a, b, c, d)|   |             | (f, a, b, c)|   |   |           |
        """
        ax = axis if axis >= 0 else ndim + axis
        if ind_axis is None:
            return ax, None
        ia = ind_axis if ind_axis >= 0 else ndim + ind_axis

        if ia > ax:
            ia -= 1

        return ax, ia


class Bank:
    """
    A feature bank object for ease in creating a table or pipeline of features to be computed.

    Parameters
    ----------
    bank_file : {None, path-like}, optional
        Path to a saved bank file to load. Optional

    Examples
    --------
    """

    __slots__ = ("_feats", "_indices")

    def __str__(self):
        return "Bank"

    def __repr__(self):
        s = "Bank["
        for f in self._feats:
            s += f"\n\t{f!r},"
        s += "\n]"
        return s

    def __contains__(self, item):
        return item in self._feats

    def __len__(self):
        return len(self._feats)

    def __init__(self, bank_file=None):
        # initialize some variables
        self._feats = []
        self._indices = []

        if bank_file is not None:
            self.load(bank_file)

    def add(self, features, index=None):
        """
        Add a feature or features to the pipeline.

        Parameters
        ----------
        features : {Feature, list}
            Single signal Feature, or list of signal Features to add to the feature Bank
        index : {int, slice, list}, optional
            Index to be applied to data input to each features. Either a index that will
            apply to every feature, or a list of features corresponding to each feature being
            added.
        """
        if isinstance(features, Feature):
            if features in self:
                warn(
                    f"Feature {features!s} already in the Bank, will be duplicated.",
                    UserWarning,
                )
            self._indices.append(partial_index_check(index))
            self._feats.append(features)
        elif all([isinstance(i, Feature) for i in features]):
            if any([ft in self for ft in features]):
                warn("Feature already in the Bank, will be duplicated.", UserWarning)
            self._indices.extend(normalize_indices(len(features), index))
            self._feats.extend(features)

    def save(self, file):
        """
        Save the feature Bank to a file for a persistent object that can be loaded later to create
        the same Bank as before

        Parameters
        ----------
        file : path-like
            File to be saved to. Creates a new file or overwrites an existing file.
        """
        out = []
        for i, ft in enumerate(self._feats):
            idx = "Ellipsis" if self._indices[i] is Ellipsis else self._indices[i]
            out.append(
                {ft.__class__.__name__: {"Parameters": ft._params, "Index": idx}}
            )

        with open(file, "w") as f:
            json.dump(out, f)

    def load(self, file):
        """
        Load a previously saved feature Bank from a json file.

        Parameters
        ----------
        file : path-like
            File to be read to create the feature Bank.
        """
        # the import must be here, otherwise a circular import error occurs
        from skdh.features import lib

        with open(file, "r") as f:
            feats = json.load(f)

        for ft in feats:
            name = list(ft.keys())[0]
            params = ft[name]["Parameters"]
            index = ft[name]["Index"]
            if index == "Ellipsis":
                index = Ellipsis

            # add it to the feature bank
            self.add(getattr(lib, name)(**params), index=index)

    def compute(
        self, signal, fs=1.0, *, axis=-1, index_axis=None, indices=None, columns=None
    ):
        """
        Compute the specified features for the given signal

        Parameters
        ----------
        signal : {array-like}
            Array-like signal to have features computed for.
        fs : float, optional
            Sampling frequency in Hz. Default is 1Hz
        axis : int, optional
            Axis along which to compute the features. Default is -1.
        index_axis : {None, int}, optional
            Axis corresponding to the indices specified in `Bank.add` or `indices`. Default is
            None, which assumes that this axis is not part of the signal. Note that setting this to
            None means values for `indices` or the indices set in `Bank.add` will be ignored.
        indices : {None, int, list-like, slice, ellipsis}, optional
            Indices to apply to the input signal. Either None, a integer, list-like, slice to apply
            to each feature, or a list-like of lists/objects with a 1:1 correspondence to the
            features present in the Bank. If provided, takes precedence over any values given in
            `Bank.add`. Default is None, which will use indices from `Bank.add`.
        columns : {None, list}, optional
            Columns to use if providing a dataframe. Default is None (uses all columns).

        Returns
        -------
        feats : numpy.ndarray
            Computed features.
        """
        # standardize the input signal
        if isinstance(signal, DataFrame):
            columns = columns if columns is not None else signal.columns
            x = signal[columns].values.astype(float64)
        else:
            try:
                x = asarray(signal, dtype=float64)
            except ValueError as e:
                raise ArrayConversionError("Error converting signal to ndarray") from e

        axis, index_axis = normalize_axes(x.ndim, axis, index_axis)

        if index_axis is None:
            indices = [...] * len(self)
        else:
            if indices is None:
                indices = self._indices
            else:
                indices = normalize_indices(len(self), indices)

        # get the number of features that will results. Needed to allocate the feature array
        if index_axis is None:
            # don't have to move any other axes than the computation axis
            x = moveaxis(x, axis, -1)
            # number of feats is 1 per
            n_feats = [1] * len(self)
            feats = zeros((sum(n_feats),) + x.shape[:-1], dtype=float64)
        else:
            # move both the computation and index axis. do this in two steps to allow for undoing
            # just the index axis swap later. The index_axis has been adjusted appropriately
            # to match this axis move in 2 steps
            x = moveaxis(x, axis, -1)
            x = moveaxis(x, index_axis, 0)

            n_feats = []
            for ind in indices:
                n_feats.append(get_n_feats(x.shape[0], ind))

            feats = zeros((sum(n_feats),) + x.shape[1:-1], dtype=float64)

        feat_i = 0  # keep track of where in the feature array we are
        for i, ft in enumerate(self._feats):
            feats[feat_i : feat_i + n_feats[i]] = ft.compute(
                x[indices[i]], fs=fs, axis=-1
            )

            feat_i += n_feats[i]

        # Move the shape back to the correct one.
        # only have to do this if there is an index axis, because otherwise the array is still in
        # the same order as originally
        if index_axis is not None:
            feats = moveaxis(feats, 0, index_axis)  # undo the previous swap/move

        return feats


class Feature(ABC):
    """
    Base feature class
    """

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        s = self.__class__.__name__ + "("
        for p in self._params:
            s += f"{p}={self._params[p]!r}, "
        if len(self._params) > 0:
            s = s[:-2]
        return s + ")"

    def __eq__(self, other):
        if isinstance(other, type(self)):
            # double check the name
            eq = str(other) == str(self)
            # check the parameters
            eq &= other._params == self._params
            return eq
        else:
            return False

    __slots__ = ("_params",)

    def __init__(self, **params):
        self._params = params

    @abstractmethod
    def compute(self, signal, fs=1.0, *, axis=-1):
        """
        Compute the signal feature.

        Parameters
        ----------
        signal : array-like
            Signal to compute the feature over.
        fs : float, optional
            Sampling frequency in Hz. Default is 1.0
        axis : int, optional
            Axis over which to compute the feature. Default is -1 (last dimension)

        Returns
        -------
        feat : numpy.ndarray
            ndarray of the computed feature
        """
        # move the computation axis to the end
        return moveaxis(asarray(signal, dtype=float64), axis, -1)
