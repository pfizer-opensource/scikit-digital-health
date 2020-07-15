"""
Core functionality for feature computation

Lukas Adamowicz
Pfizer DMTI 2020
"""
from numpy import ndarray, array, zeros, sum
from pandas import DataFrame

from PfyMU.features.utility import standardize_signal, compute_window_samples


__all__ = ['Bank']


class NotAFeatureError(Exception):
    """
    Custom error for indicating an attempt to add something that is not a feature to a features.Bank
    """
    pass


class NoFeaturesError(Exception):
    """
    Custom error if there are no features in the feature Bank
    """
    pass


class Bank:
    __slots__ = ('_feat_list', '_n_feats', '_eq_idx', 'wlen_s', 'wstep')

    def __str__(self):
        ret = '[\n'
        for i in self._feat_list:
            ret += f'\t{i.parent.__str__()},\n'
        ret += ']'
        return ret

    def __init__(self, window_length=None, window_step=1.0):
        """
        A feature bank for ease in creating a table of features for a given signal, applying the windowing as specified.

        Parameters
        ----------
        window_length : float
            Window length in seconds. If not provided (None), will do no windowing. Default is None
        window_step : {float, int}
            Window step - the spacing between the start of windows. This can be specified several different ways
            (see Notes). Default is 1.0

        Notes
        -----
        Computation of the window step depends on the type of input provided, and the range.
        - `window_step` is a float in (0.0, 1.0]: specifies the fraction of a window to skip to get to the start of the
        next window
        - `window_step` is an integer > 1: specifies the number of samples to skip to get to the start of the next
        window


        Examples
        --------
        >>> fb = Bank()
        >>> # add features to the Bank
        >>> fb + Mean()
        >>> fb + Range()
        >>> # add specific axes of features to the Bank
        >>> fb + SignalEntropy()[0]
        >>> fb + IQR()[[1, 2]]
        >>> fb + SignalEntropy()[2]  # this will reuse the same instance created above, to avoid feature re-computation
        >>> features = fb.compute(signal)
        """
        # storage for the features to calculate
        self._feat_list = []
        # storage for the number of features that will be calculated
        self._n_feats = None  # need to allocate in compute to reset for each compute call
        # storage of the last instance of a particular class/instance, if it exists
        self._eq_idx = None

        # windowing parameters
        self.wlen_s = window_length
        self.wstep = window_step

    def compute(self, signal, fs=None, columns=None, windowed=False):
        """
        Compute the features in the Bank.

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D), or a pandas.DataFrame containing the signal to be analyzed
        fs : float, optional
            Sampling frequency of the signal in Hz. Only required if the features in the Bank require
            sampling frequency in the computation (see feature documentation), or if windowing `signal`.
        columns : array-like, optional
            Columns to use from the pandas.DataFrame. If signal is an ndarray, providing columns will provide
            a return of the column/feature name combinations that matches the columns in the returned ndarray
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        features : {numpy.ndarray, pandas.DataFrame}
            Table of computed features in the same type as the input.
        """
        if not self._feat_list:
            raise NoFeaturesError('No features to compute.')

        self._n_feats = []

        # compute windowing # of samples if necessary
        if self.wlen_s is not None and self.wstep is not None:
            window_length, window_step = compute_window_samples(fs, self.wlen_s, self.wstep)
        else:
            window_length, window_step = None, None

        # standardize the input signal, and perform windowing if desired
        x, columns = standardize_signal(signal, windowed=windowed, window_length=window_length,
                                        step=window_step, columns=columns)
        feat_columns = []  # allocate if necessary

        # ensure if passing an ndarray, that the columns matches the appropriate shape
        if not isinstance(signal, DataFrame):
            if columns is not None:
                if len(columns) != x.shape[-1]:
                    raise ValueError(f'Provided column names ({len(columns)}) does not match the number of columns'
                                     f'in the data ({x.shape[-1]}).')

        # first get the number of features expected so the space can be allocated
        for dft in self._feat_list:
            # need this if statement to deal with ellipsis indices
            if dft.n == -1:
                dft.n = x.shape[-1]  # number of axes is last

            self._n_feats.append(dft.n)

        # allocate the feature table. This accounts for multiple columns per feature
        feats = zeros((x.shape[0], sum(self._n_feats)))

        idx = 0  # set a counter to keep track of where to put each computed feature

        # iterate over all the features and compute them, saving the result as desired in the feature table
        for i, dft in enumerate(self._feat_list):
            dft._compute(x, fs)  # compute the feature without returning it

            feats[:, idx:idx + self._n_feats[i]] = dft.get_result()  # get the result
            if isinstance(signal, DataFrame):
                feat_columns.append(dft.get_columns(columns))
            elif columns is not None:
                feat_columns.append(dft.get_columns(columns))

            idx += self._n_feats[i]  # increment the index tracker

        if isinstance(signal, ndarray):
            if columns is not None:
                return feats, feat_columns
            else:
                return feats
        elif isinstance(signal, DataFrame):
            return DataFrame(data=feats, columns=feat_columns)

    # FUNCTIONALITY METHODS
    def __contains__(self, item):
        isin = False

        if isinstance(item, (Feature, DeferredFeature)):
            for i, dft in enumerate(self._feat_list):
                comp = item == dft
                isin |= comp

                # save the location of the last equivalent item
                if comp:
                    self._eq_idx = i

        return isin

    def __add__(self, other):
        if isinstance(other, Feature):
            self._feat_list.append(DeferredFeature(other, ...))
        elif isinstance(other, DeferredFeature):
            if other in self:
                self._feat_list.append(DeferredFeature(self._feat_list[self._eq_idx].parent, other.index))
            else:
                self._feat_list.append(other)
        else:
            raise NotAFeatureError(f'Cannot add an object of type ({type(other)}) to a feature Bank.')


class Feature:
    def __str__(self):
        s = ''
        for key in self._eq_params:
            if isinstance(self._eq_params[key], float):
                s += f'{key}={self._eq_params[key]:.2f}'
            else:
                s += f'{key}={self._eq_params[key]}, '
        s = s[:-2]
        return f'{self._name}({s})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, name, eq_params):
        """
        Base feature class. intended to be overwritten

        Parameters
        ----------
        name : str
            Feature name. Used for creating the str/repr of the feature
        eq_params : dict
            Dictionary of parameter names and their values. Used for creating the str/repr of the feature, and
            checking equivalence between features
        """
        self._x = None
        self._result = None

        # mappings for indices
        self._xyz_map = {'x': 0, 'y': 1, 'z': 2}
        self._xyz_comp_map = {'xy': 0, 'xz': 1, 'yz': 2}

        # parameters/feature information
        self._name = name
        self._eq_params = eq_params

    # PUBLIC METHODS
    def compute(self, signal, fs=None, *, columns=None, windowed=False):
        """
        Compute the feature.

        Parameters
        ----------
        signal : {numpy.ndarray, pandas.DataFrame}
            Either a numpy array (up to 3D) or a pandas dataframe containing the signal
        fs : float, optional
            Sampling frequency in Hz
        columns : array_like, optional
            Columns to use if signal is a pandas.DataFrame. If None, uses all columns.
        windowed : bool, optional
            If the signal has already been windowed. Default is False.

        Returns
        -------
        feature : {numpy.ndarray, pandas.DataFrame}
            Computed feature, returned as the same type as the input signal
        """
        # check fs
        if fs is not None:
            if not isinstance(fs, (float, int)):
                raise ValueError("fs must be a float or int. If trying to specify columns, it is keyword-required")

        # set the result to None, to force re-computation. publicly should always be re-computing. The benefit
        # for avoiding re-computation comes for the feature Bank pipeline of computation, where the same result
        # might be used multiple times but for different indices
        self._result = None

        # extract and standardize the data. No windowing in the public method here
        x, columns = standardize_signal(signal, windowed=windowed, window_length=None, step=None, columns=columns)

        self._compute(x, fs)

        if isinstance(signal, ndarray):
            return self._result
        elif isinstance(signal, DataFrame):
            return DataFrame(data=self._result, columns=[f'{self._name}_{i}' for i in columns])

    # PRIVATE METHODS
    def _compute(self, x, fs):
        # if the result is already defined, don't need to compute again. Note that if calling from the public
        # Feature.compute() method, _result is automatically set to None for re-computation each time it is called
        if self._result is not None:
            return

    # FUNCTIONALITY METHODS
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (other._eq_params == self._eq_params) and (other._name == self._name)
        elif isinstance(other, DeferredFeature):
            return (other.parent._eq_params == self._eq_params) and (other.parent._name == self._name)
        else:
            return False

    def __getitem__(self, key):
        index = None
        if isinstance(key, str):
            index = self._xyz_map.get(key, None)  # default return is None if doesn't exist
            if index is None:
                index = self._xyz_comp_map.get(key, None)
        elif isinstance(key, (int, type(Ellipsis))):
            index = key
        elif isinstance(key, (list, tuple, ndarray)):
            if key[0] in self._xyz_comp_map:
                index = [self._xyz_comp_map[i] for i in key]
            elif key[0] in self._xyz_map:
                index = [self._xyz_map[i] for i in key]
            else:
                index = key

        if index is None:
            raise IndexError("Index must be a int, in ['x', 'y', 'z', 'xy', 'xz', 'yz'] or an array-like of those")

        return DeferredFeature(self, index)


class DeferredFeature:
    __slots__ = ('parent', 'index', '_compute', 'n')  # limit attributes

    def __str__(self):
        return f'Deferred{self.parent._name}'

    def __repr__(self):
        return f'Deferred{self.parent.__repr__()}'

    def __init__(self, parent, index):
        """
        An object for storing a feature for deferred computation. Stores the parent feature, as well as the desired
        index to return of the results

        Parameters
        ----------
        parent : Feature
            Parent feature which contains the computations for feature calculation
        index : int, array-like of ints, Ellipsis
            Index to retrieve for the results
        """
        if isinstance(parent, Feature):
            self.parent = parent
        else:
            raise NotAFeatureError('DeferredFeature parent must be a Feature or subclass')

        self.index = index
        # we want the "private" method that doesn't return a value, and checks if the computation was done already
        self._compute = self.parent._compute

        # determine how many features will be returned
        if hasattr(index, "__len__"):
            self.n = len(index)
        elif isinstance(index, type(Ellipsis)):  # can't figure this out until later, when we know signal size
            self.n = -1
        else:
            self.n = 1  # if not an array-like, only returning 1 value

    def get_result(self):
        return self.parent._result[:, self.index]

    def get_columns(self, columns):
        return [f'{i}_{self.parent._name.lower()}' for i in array(columns)[self.index]]

    # FUNCTIONALITY METHODS
    def __eq__(self, other):
        if isinstance(other, DeferredFeature):
            return (other.parent._eq_params == self.parent._eq_params) and (other.parent._name == self.parent._name)
        elif isinstance(other, Feature):
            return (other._eq_params == self.parent._eq_params) and (other._name == self.parent._name)
        else:
            return False






