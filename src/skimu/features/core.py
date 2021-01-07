"""
Core functionality for feature computation
Lukas Adamowicz
Pfizer DMTI 2020
"""
from collections.abc import Iterator, Sequence
import json
from warnings import warn

from pandas import DataFrame


__all__ = ["Bank"]


def partial_index_check(index):
    if isinstance(index, str) or isinstance(index, float):
        raise IndexError("Indices cannot be strings or floats.")
    if index is None:
        index = ...
    return index


def normalize_axes(ndim, is_df, axis, ind_axis):
    """
    Normalize input axes to be positive/correct for how the swapping has to work
    """
    if axis == ind_axis:
        raise IndexError("axis and index_axis cannot be the same")

    if is_df:
        return 0, 0  # has to be 0 for the way the double swap works
    elif ndim == 1:
        return 0, None
    elif ndim >= 2:
        """
        |  shape | ax | ia |  swap1 | ax | ia |  res  | ax | ia | res swap |
        |--------|----|----|--------|----|----|-------|----|----|----------|
        | (a, b) |  0 |  1 | (b, a) |  0 |  0 | (bf,) |    |    |          |
        | (a, b) |  0 |  N | (b, a) |  0 |  N | (f, b)|    |    |          |
        | (a, b) |  1 |  0 |        |    |    | (3a,) |    |    |          |
        | (a, b) |  1 |  N |        |    |    | (f, a)|    |    |          |

        |  shape   | ax| ia|   swap1  | ax| ia|   swap2  | ax | ia |   res    | ia |   | res swap |
        |----------|---|---|----------|---|---|----------|----|----|----------|----|---|----------|
        | (a, b, c)| 0 | 1 | (c, b, a)| 2 | 1 | (b, c, a)|  2 |  0 | (bf, c)  |  0 | 0 | (bf, c)  |
        | (a, b, c)| 0 | 2 | (c, b, a)| 2 | 0 |          |    |    | (cf, b)  |  0 | 1 | (b, cf)  |
        | (a, b, c)| 0 | N | (c, b, a)| 2 | N |          |    |    | (f, c, b)|  1 | 2 | (f, b, c)|
        | (a, b, c)| 1 | 0 | (a, c, b)| 2 | 0 |          |    |    | (af, c)  |  0 | 0 | (af, c)  |
        | (a, b, c)| 1 | 2 | (a, c, b)| 2 | 1 | (c, a, b)|  2 |  0 | (cf, a)  |  0 | 1 | (a, cf)  |
        | (a, b, c)| 1 | N | (a, c, b)| 2 | N |          |    |    | (f, a, c)|  1 | 1 | (f, a, c)|
        | (a, b, c)| 2 | 0 |          |   |   |          |    |    | (af, b)  |  0 | 0 | (af, b)  |
        | (a, b, c)| 2 | 1 |          |   |   | (b, a, c)|  2 |  0 | (bf, a)  |  0 | 1 | (a, bf)  |
        | (a, b, c)| 2 | N |          |   |   |          |    |    | (f, a, b)|  1 | 1 | (f, a, b)|
        """
        ax = axis if axis >= 0 else ndim + axis
        if ind_axis is None:
            return ax, None
        ia = ind_axis if ind_axis >= 0 else ndim + ind_axis

        if ia == ndim - 1:
            ia = ax  # set to the axis, since this will be in the correct position after first swap

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
        features : {Feature, list-like}
            Single signal Feature, or list of signal Features to add to the feature Bank
        index : {int, slice, list}, optional
            Index to be applied to data input to each features. Either a index that will
            apply to every feature, or a list of features corresponding to each feature being
            added.
        """
        if isinstance(features, Feature):
            if features in self:
                warn(f"Feature {features!s} already in the Bank, will be duplicated.")
            self._idnices.append(partial_index_check(index))
            self._feats.append(features)
        elif all([isinstance(i, Feature) for i in features]):
            if index is None:
                for ft in features:
                    self._indices.append(...)
                    self._feats.append(ft)
            elif not isinstance(index, Iterator):  # slice, single integer, etc
                for ft in features:
                    self._indices.append(partial_index_check(index))
                    self._feats.append(ft)
            elif all([isinstance(i, int) for i in index]):  # iterable of ints
                for ft in features:
                    self._indices.append(index)
                    self._feats.append(ft)
            elif isinstance(index, Sequence):  # able to be indexed
                for i, ft in enumerate(features):
                    self._indices.append(partial_index_check(index[i]))
                    self._feats.append(ft)
            else:
                raise IndexError("Index not understood.")

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
        for ft in self._feats:
            idx = "Ellipsis" if ft.index is Ellipsis else ft.index
            out.append(
                {
                    ft.__class__.__name__: {
                        "Parameters": ft._params,
                        "Index": idx
                    }
                }
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
        from skimu.features import lib

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

    def compute(self, signal, fs=1., *, axis=-1, index_axis=None, indices=None):
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
            None means values for `indices` or the indices set in `Bank.add` will be ignored
        indices : {None, int, list-like, slice, ellipsis}, optional
            Indices to apply to the input signal. Either a integer, list-like, slice to apply to
            each feature, or a list-like of lists/objects with a 1:1 correspondence to the
            features present in the Bank.

        Returns
        -------
        feats : numpy.ndarray
            Computed features.
        """
        axis, index_axis = normalize_axes(
            signal.ndim, isinstance(signal, DataFrame), axis, index_axis)

