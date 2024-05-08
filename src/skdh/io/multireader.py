from collections.abc import Mapping
from warnings import warn

from numpy import (
    argsort,
    concatenate,
    mean,
    diff,
    median,
    allclose,
    argmin,
    argmax,
    nonzero,
    ndarray,
)

from skdh.base import BaseProcess, handle_process_returns
from skdh import io
from skdh.utility.internal import apply_resample


class MultiReader(BaseProcess):
    """
    A process for reading in multiple files into one set of datastreams for processing.

    Parameters
    ----------
    mode : {'combine', 'concatenate', 'leave'}
        The mode to use when reading multiple files. Options are "combine", which
        combines multiple data-streams from different files, or "concatenate" which
        combines the same data-stream from multiple files (Case 3 `reader_kw` not
        allowed). Finally, "leave" is only valid when providing case 3 for `reader_kw`
        (see Notes), and leaves the results in separate dictionaries with titles
        given by the keys of `reader_kw` and `files` as passed to the `predict` method.
    reader : str
        The name of the reader class to use. See :ref:`SKDH IO`.
    reader_kw : {None, array-like, dict}, optional
        Reader key-word arguments to initialize the reader. See Notes for 3 specification
        options.
    resample_to_lowest : bool, optional
        When `mode` is "combine", resample separate datastreams to the lowest sampled
        stream. Default is True. False will re-sample all datastreams to the highest
        sampled stream.

    Notes
    -----
    The combine `mode` should be used in the case when you have, for example, acceleration
    and temperature in separate files. Concatenate should be used when multiple files
    all contain acceleration, for example. In this case, read results will be concatenated
    together in order of the first timestamp.

    `reader_kw` has three(four) different ways to be specified:

    0. None for no keyword arguments overriding defaults
    1. A dictionary of key-word arguments that will be the same for all files.
    2. A list of dictionaries of key-word arguments that must equal the number of
        files provided to the `predict` method, in order.
    3. A dictionary of dictionaries of key-word arguments. In this case, the `files`
        argument for `predict` should be a dictionary with the same key-names as
        `reader_kw`, and the key-word arguments will be associated with the path
        in the `files` dictionary.

    Note that if the `reader` returns the same keys, and `mode` is "combine",
    keys will be overwritten.

    Examples
    --------
    Case 0:

    >>> mrdr = MultiReader(mode='combine', reader='ReadCsv', reader_kw=None)
    >>> mrdr.predict(files=["file1.csv", "file2.csv"])

    Case 1:

    >>> kw = {'time_col_name': 'timestamp', 'read_csv_kwargs': {'skiprows': 5}}
    >>> mrdr = MultiReader(mode='combine', reader='ReadCsv', reader_kw=kw)
    >>> mrdr.predict(files=["file1.csv", "file2.csv"])

    Case 2:

    >>> kw = [
    >>>     {'time_col_name': 'ts', 'column_names': {'accel': ['ax', 'ay', 'az']}},
    >>>     {'time_col_name': 'ts', 'column_names': {'temperature': 'temperature C'}},
    >>> ]
    >>> mrdr = MultiReader(mode='combine', reader='ReadCsv', reader_kw=kw)
    >>> mrdr.predict(files=["file1.csv", "file2.csv"])

    Case 3:

    >>> kw = {
    >>>     'f1': {'time_col_name': 'ts', 'column_names': {'accel': ['ax', 'ay', 'az']}},
    >>>     'f2': {'time_col_name': 'ts', 'column_names': {'temperature': 'temperature C'}},
    >>> }
    >>> mrdr = MultiReader(mode='combine', reader='ReadCsv', reader_kw=kw)
    >>> mrdr.predict(files={'f1': "file1.csv", 'f2': "file2.csv"})
    """

    def __init__(self, mode, reader, reader_kw=None, resample_to_lowest=True):
        super().__init__(
            mode=mode, reader=reader, resample_to_lowest=resample_to_lowest
        )

        if reader_kw is None:
            reader_kw = {}

        if mode.lower() in ["leave", "combine", "concatenate"]:
            self.mode = mode.lower()
        else:
            raise ValueError("mode must be one of {'leave', 'combine', 'concatenate'}.")

        # get the reader classes
        self.rdr = getattr(io, reader)
        self.reader_kw = reader_kw

        self.resample_to_lowest = resample_to_lowest

    def get_reader_kw(self, idx):
        """
        Get the appropriate reader class key-word arguments

        Parameters
        ----------
        idx : str, int
            Index of the file to use to retrieve reader kwargs
        """
        try:
            # case 0: reader_kw == None -> reader_kw = {} -> [str, int] KeyError
            # case 1: reader_kw == {...} -> [str, int] KeyError
            # case 2: reader_kw == [] -> [str] TypeError [int] IndexError
            # case 3: reader_kw == {...} -> [str, int] KeyError
            return self.reader_kw[idx]
        except (TypeError, KeyError):
            return self.reader_kw
        except IndexError:
            raise IndexError("More files provided than reader key-word arguments.")

    def handle_combine(self, res):
        """
        Combine results

        Parameters
        ----------
        res : {list, dict}

        Returns
        -------
        results : dict
            Datastream results dictionary
        """
        if isinstance(res, dict):
            res = [
                v for _, v in res.items()
            ]  # standardize since we dont care about labels

        # get the last time available
        t_ends = [i["time"][-1] for i in res]
        t_end = min(t_ends)

        # check if we need to resample time streams
        all_fs = [i.get("fs", 1 / mean(diff(i["time"][:5000]))) for i in res]
        # allow 0.1 absolute tolerance to account for hand-calculation of fs
        needs_resample = not allclose(all_fs, median(all_fs), atol=0.1)

        if needs_resample:
            # get the index we want to resample to
            resample_idx_fn = {True: argmin, False: argmax}[self.resample_to_lowest]
            resample_idx = resample_idx_fn(all_fs)

            # get the base results dictionary, and remove from the rest of the results
            # so that we can iterate over the rest and resample them
            results = res.pop(resample_idx)
            # get the length of the acceptable data so that it all matches
            n = nonzero(results["time"] <= t_end)[0][-1] + 1
            for k, v in results.items():
                if isinstance(v, ndarray):
                    results[k] = v[:n]

            time_rs = results["time"]

            for rdict in res:
                # deconstruct data to resample
                time = rdict.pop("time")
                keys = []
                data = ()
                for key, val in rdict.items():
                    if isinstance(val, ndarray):
                        keys.append(key)
                        data += (val,)
                # drop the keys
                for k in keys:
                    rdict.pop(k)

                _, data_rs, *_ = apply_resample(
                    time=time,
                    time_rs=time_rs,
                    data=data,
                    aa_filter=True,
                )

                # update results with re-sampled data, and any remaining keys in the dictionary
                # that aren't file and fs. File will get overwritten later
                rdict.pop("fs", None)

                for k, v in zip(keys, data_rs):
                    if k in results:
                        warn(
                            f"Data {k} is already in the results when combining, overwriting."
                        )
                    results[k] = v
                results.update(rdict)
        else:
            # if no resampling needed, just update/combine all the dictionaries
            results = res[0]
            for rdict in res[1:]:
                results.update(rdict)

        return results

    @staticmethod
    def concat(data):
        """
        Custom concatenation function for data streams to handle inputs that are
        either tuples of arrays or tuples of dictionaries

        Parameters
        ----------
        data : tuple
            Tuple of either numpy.ndarrays to concatenate, or dictionaries whose
            keys should be concatenated.

        Returns
        -------
        data : {ndarray, dict}
            Concatenated data.
        """
        if all([isinstance(i, ndarray) for i in data]):
            return concatenate(data, axis=0)
        elif all([isinstance(i, dict) for i in data]):
            res = {}
            for k in data[0]:
                if isinstance(data[0][k], (ndarray, list)):
                    res[k] = concatenate([d[k] for d in data], axis=0)
                else:  # this might cause issues later on, but for now leave it
                    res[k] = data[0][k]
            return res
        else:
            raise ValueError(
                "Data to be concatenated must be either all numpy.ndarrays or all dictionaries."
            )

    def handle_concatenation(self, res):
        """
        Concatenate results.

        Parameters
        ----------
        res : list
            List of results dictionaries

        Returns
        -------
        results : dict
            Dictionary of results datastreams
        """
        if isinstance(res, dict):
            res = [
                v for _, v in res.items()
            ]  # standardize since we dont care about labels

        t0s = [i["time"][0] for i in res]
        # get sorted index
        t0_isort = argsort(t0s)

        results = {}
        res_lists = {}
        # split between concatenatable items and non
        for k, v in res[0].items():
            if isinstance(v, (ndarray, dict)):
                res_lists[k] = []
            else:
                results[k] = v

        # get items in time order
        for ires in t0_isort:
            res_dict = res[ires]

            for k in res_lists:
                try:
                    res_lists[k].append(res_dict.pop(k))
                except KeyError:
                    raise KeyError(
                        "To concatenate file contents, all files must have the same data streams."
                    )
            # update the non-concatenatable items results
            results.update(res_dict)

        results.update({k: self.concat(v) for k, v in res_lists.items()})

        return results

    def handle_results(self, res):
        """
        Handle the combining of results.

        Parameters
        ----------
        res : {dict, list}
            Dictionary or list of results.

        Returns
        -------
        results : dict
            Dictionary of final results
        """
        if isinstance(res, dict):
            if self.mode == "leave":
                return res
            elif self.mode == "combine":
                results = self.handle_combine(res)
            elif self.mode == "concatenate":
                results = self.handle_concatenation(res)
        else:  # res is a list
            if self.mode == "combine":
                results = self.handle_combine(res)
            elif self.mode == "concatenate":
                results = self.handle_concatenation(res)
            else:
                raise ValueError(
                    "Only {combine, concatenate} are valid for list specified files."
                )

        return results

    @handle_process_returns(results_to_kwargs=True)
    def predict(self, *, files=None, **kwargs):
        """
        predict(*, files=None, **kwargs)

        Read the files or files from a directory.

        Parameters
        ----------
        files : {array-like, dict}
            Either a list-like of files to read, or a dictionary of keys corresponding
            to files to read. Keys match with those provided to `reader_kw` upon
            initializing the process.

        Notes
        -----
        Note that any additional key-word arguments passed to `MultiReader.predict`
        will be passed along to the `reader.predict` method.
        """
        i0 = 0
        if isinstance(files, Mapping):
            i0 = next(iter(files))  # get the first key
            pre_results = {}
            for fid, fpath in files.items():
                # get the kwargs
                kw = self.get_reader_kw(fid)
                pre_results[fid] = self.rdr(**kw).predict(file=fpath, **kwargs)
        else:
            pre_results = []
            for i, fpath in enumerate(files):
                kw = self.get_reader_kw(i)
                pre_results.append(self.rdr(**kw).predict(file=fpath, **kwargs))

        results = self.handle_results(pre_results)

        # handle setting the file, either the first key from a dictionary or the
        # first index from a list
        results["file"] = files[i0]

        return results
