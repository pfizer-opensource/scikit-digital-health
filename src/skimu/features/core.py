"""
Core functionality for feature computation
Lukas Adamowicz
Pfizer DMTI 2020
"""
from abc import ABC, abstractmethod
from collections.abc import Sized, Iterable
from warnings import warn
import json

from numpy import asarray, zeros, atleast_2d, sum, float_
from pandas import DataFrame

__all__ = ["Bank"]


class NotAFeatureError(Exception):
    """
    Custom error for indicating an attempt to add something that is not a feature to a
    features.Bank
    """
    pass


class NoFeaturesError(Exception):
    """
    Custom error if there are no features in the feature Bank
    """
    pass


class ArrayConversionError(Exception):
    """Custom error if input cannot be converted to an array"""
    pass


class Bank:
    """
    A feature bank for ease in creating a table or pipeline of features to be computed.

    Examples
    --------
    >>> fb = Bank()
    >>> # add features to the bank
    >>> fb.add([Mean(), Range()])
    >>> # add specific axes to the bank
    >>> fb.add([
    >>>     SignalEntropy()[0],
    >>>     IQR(),
    >>>     SignalEntropy()[1:]
    >>> ])
    >>> features = fb.compute(signal, axis=1, col_axis=0)
    """
    __slots__ = "_feat_list", "_n_feats", "_eq_idx"

    def __str__(self):
        return "Bank"

    def __repr__(self):
        ret = "[\n"
        for i in self._feat_list:
            ret += f"\t{i.parent!r},\n"
        ret += "]"
        return ret

    def __init__(self):
        # storage for the features to calculate
        self._feat_list = []
        # storage for the number of features that will be calculated
        self._n_feats = None  # need to allocate in compute to reset for each compute call
        # storage of the last instance of a particular class/instance, if it exists
        self._eq_idx = None

    # FUNCTIONALITY
    def __contains__(self, item):
        isin = False

        if isinstance(item, Feature):
            for ft in self._feat_list:
                isin |= (item == ft)

        return isin

    def __len__(self):
        return len(self._feat_list)

    # ADDING features
    def add(self, features):
        if isinstance(features, Feature):
            if features in self:
                warn(
                    f"Feature {features.__class__.__name__} already in the Bank. Adding again.",
                    UserWarning
                )
            self._feat_list.append(features)
        elif isinstance(features, (Sized, Iterable)):
            if all(isinstance(i, Feature) for i in features):
                if any(i in self for i in features):
                    warn("Repeated features. Adding again.")
                self._feat_list.extend(features)
            else:
                raise NotAFeatureError(
                    f"One of the features is not a skimu.features.Feature subclass")
        else:
            raise NotAFeatureError(
                f"Cannot add objects that are not skimu.features.Feature subclasses")

    # SAVING AND LOADING METHODS
    def save(self, file):
        """
        Save the features in the feature bank to a JSON file for easy (re-)creation of the Bank

        Parameters
        ----------
        file : {str, Path}
            File path to save to.
        """
        out = []

        for ft in self._feat_list:
            idx = "Ellipsis" if ft.index is Ellipsis else ft.index
            out.append(
                {
                    ft.__class__.__name__: {
                        "Parameters": ft._params,
                        "index": idx
                    }
                }
            )

        with open(file, "w") as f:
            json.dump(out, f)

    def load(self, file):
        """
        Load a set of features from a JSON file

        Parameters
        ----------
        file : {str, Path}
            File path to load from. File should be the output of `Bank.save(file)`
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
            self.add(getattr(lib, name)(**params)[index])

    # COMPUTATION
    def compute(self, signal, fs=None, *, axis=-1, col_axis=-2, columns=None):
        """
        Compute the features in the Bank

        Parameters
        ----------
        signal : array-like
            The signal to be processed
        fs : float, optional
            Sampling frequency in Hz. Optional. Default is None
        axis : int
            Axis along which the features are computed. Default is the last axis (-1)
        col_axis : int
            Axis which corresponds to the columns. Default is second to last axis (-2). If signal
            is 2D, this is automatically inferred to be the other dimension.
        columns : array-like
            Array of column names. If passing a pandas.DataFrame, this will be used
            to select specific columns. If passing other types of data, this will be used
            to generate feature/column name combinations

        Returns
        -------
        features : numpy.ndarray
            Array of computed features
        feat_names : array-like, optional
            Feature and column names merged. Only returned if `signal` was a pandas.DataFrame
            or columns was provided.

        Raises
        ------
        ArrayConversionError
            If `signal` cannot be converted to a numpy.ndarray
        ValueError
            If the number of provided column names doesn't match the shape of the columns axis
        """

        # make sure col_axis and axis are not the same
        if col_axis == axis:
            raise ValueError("column_axis and axis cannot be the same")

        # standardize the input signal
        if isinstance(signal, DataFrame):
            if columns is not None:
                x = atleast_2d(signal[columns].values.astype(float_))
            else:
                x = atleast_2d(signal.values.astype(float_))
                columns = signal.columns
            # set axis/col_axis
            axis = 0
            col_axis = 1
        else:
            try:
                x = atleast_2d(asarray(signal, dtype=float_))
            except ValueError as e:
                raise ArrayConversionError(
                    f"signal ({type(signal)}) cannot be converted to an array") from e

            # standardize the axis and column axis arguments
            axis = axis if axis >= 0 else x.ndim + axis
            col_axis = col_axis if col_axis >= 0 else x.ndim + col_axis
            # if 2d, get the col_axis based on axis
            col_axis = col_axis if x.ndim > 2 else 1 - axis

        # make sure columns matches the shape
        if columns is not None:
            if len(columns) != x.shape[col_axis]:
                raise ValueError("Provided columns does not match signal col_axis shape.")

        # get the number of features expected so the space can be allocated
        n_feats = []
        for ft in self._feat_list:
            # need this if statement do deal with unkown # of indices
            if ft.n == -1:
                if isinstance(ft.index, slice):
                    n_feats.append(len(range(*ft.index.indices(x.shape[col_axis]))))
                else:
                    n_feats.append(x.shape[col_axis])
            else:
                n_feats.append(ft.n)

        # move the axis array to the end (for C-storage when copied in extensions)
        x = x.swapaxes(axis, -1)
        x = x.swapaxes(col_axis, 0)  # move column axis first for easy indexing
        shape = x.shape
        shape[0] = sum(n_feats)

        feats = zeros(shape[:-1])
        feat_cols = [] if columns is not None else None

        idx = 0  # keep track of where the features are being inserted

        for i, ft in enumerate(self._feat_list):
            feats[idx:idx + n_feats[i]] = ft._compute(x[ft.index], fs)

            if feat_cols is not None:
                feat_cols.extend(ft._get_cols(columns))

            idx += n_feats[i]
        # swap back the features
        feats = feats.swapaxes(col_axis, 0)

        if feat_cols is not None:
            return feats, feat_cols
        else:
            return feats


class Feature(ABC):
    """
    Base feature class. Intended to be overwritten

    Parameters
    ----------
    **params : any
        Any parameters that are passed to the subclassed objects
    """
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        s = self.__class__.__name__ + "("
        for key in self._params:
            s += f"{key}={self._params[key]!r}, "
        s = s[:-2] + ")"
        return s

    __slots__ = "_params", "index", "n"

    def __init__(self, **params):
        super().__init__()

        self._params = params  # storing for equivalence testing
        self.index = ...  # index to compute
        """
        number of values the feature returns, which will usually depend on the input signal
        -1 : depends on input signal
        N >= 0 : fixed number of returns
        
        since the default index is ..., the returns is not determined until later
        """
        self.n = -1

    # FUNCTIONALITY methods
    def __eq__(self, other):
        if isinstance(other, type(self)):
            # are the names the same
            cond1 = other.__class__.__name__ == self.__class__.__name__
            # are the params the same
            cond2 = other._params == self._params
            # is the index the same
            cond3 = other.index == self.index
            return cond1 & cond2 & cond3
        else:
            return False

    def __getitem__(self, item):
        self.index = item
        # get the number of returns
        if isinstance(item, Sized):
            if ... in item:
                raise ValueError("Cannot use fancy indexing")
            if len(item) >= 2 and any([isinstance(i, slice) for i in item]):
                raise ValueError("Cannot use fancy indexing")
            self.n = len(item)
        elif isinstance(item, int):
            self.n = 1
        elif isinstance(item, slice):
            self.n = -1
        else:
            raise ValueError("Index not understood")

    # PUBLIC METHODS
    @abstractmethod
    def compute(self, signal, fs=1., *, axis=-1, col_axis=None, columns=None):
        """
        Compute the feature

        Parameters
        ----------
        signal : array-like
            The signal of interest
        fs : float, optional
            Sampling frequency for the signal, in Hz. Default is 1.
        axis : optional
            Axis along which the feature is computed.
        col_axis : optional
            Axis of the columns
        columns : array-like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns

        Returns
        -------
        feature : array-like
            Computed feature.
        """
        fs = float(fs)

        # make sure it is an array
        if isinstance(signal, DataFrame):
            if columns is not None:
                x = signal[columns].values.astype(float_)
            else:
                x = signal.values.astype(float_)
            axis = 0
            col_axis = 0  # needs to be the same to avoid swapping and undoing a previous swap
        else:
            x = asarray(signal, dtype=float_)
            # standardize the axis and column axis arguments
            axis = axis if axis >= 0 else x.ndim + axis
            col_axis = col_axis if col_axis >= 0 else x.ndim + col_axis
            # if 2d, get the col_axis based on axis
            col_axis = col_axis if x.ndim > 2 else 0

        if x.ndim == 1:
            res = self._compute(x, fs)
            return res

        x = x.swapaxes(axis, -1)
        x = x.swapaxes(col_axis, 0)
        res = self._compute(x[self.index], fs)
        """
        If only 1 dimension was lost, swap the column axis
        If more than 1 dimension was lost, then the indexing operation was what got rid of an axis
        so we don't need to swap anything
        
        signal.shape -> (100, 300, 3)
        axis -> 1
        col_axis -> 2
        x.shape -> (3, 100, 300)
        
        EX
        index -> 0
        x[index].shape -> (100, 300)
        res.shape -> (100,)
        
        EX2
        index -> ...
        x[index].shape -> (3, 100, 300)
        res.shape -> (3, 100)
        res.swapaxes -> (100, 3) :: matches signal shape order
        """
        if res.ndim == x.ndim - 1 and res.ndim > 1:
            res = res.swapaxes(col_axis, 0)

        return res

    # PRIVATE METHODS
    @abstractmethod
    def _compute(self, x, fs):
        pass

    def _get_columns(self, cols):
        cols = asarray(cols)[self.index]

        name = f"{self.__class__.__name__.replace(' ', '_')}"
        return [f"{name}_{col}" for col in cols]
