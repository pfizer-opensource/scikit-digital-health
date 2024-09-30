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
from skdh.utility.internal import apply_resample, fill_data_gaps


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
    fill_gaps : bool, optional
        Fill any gaps in data streams where possible (same size as time array). Default
        is True.
    fill_value : {None, dict}, optional
        Dictionary with keys and values to fill data streams with. See Notes for
        default values if not provided.
    gaps_error : {'raise', 'warn', 'ignore'}, optional
        Behavior if there are gaps in the datastreams. Default is to raise an error.
    require_all_keys : bool, optional
        Require all files to provide the same keys. Default is True.

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

    Default fill values are:

    - accel: numpy.array([0.0, 0.0, 1.0])
    - gyro: 0.0
    - temperature: 0.0
    - ecg: 0.0

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

    def __init__(
        self,
        mode,
        reader,
        reader_kw=None,
        resample_to_lowest=True,
        fill_gaps=True,
        fill_value=None,
        gaps_error="raise",
        require_all_keys=True,
        ignore_file_size_check=True,
    ):
        super().__init__(
            mode=mode,
            reader=reader,
            resample_to_lowest=resample_to_lowest,
            fill_gaps=fill_gaps,
            fill_value=fill_value,
            gaps_error=gaps_error,
            require_all_keys=require_all_keys,
            ignore_file_size_check=ignore_file_size_check,
        )

        # Set a variable here so that we can ignore file size checking when reading
        # multiple files. This should then get handled by gap filling.
        self._skip_file_size_check = True

        if reader_kw is None:
            reader_kw = {}

        if mode.lower() in ["leave", "combine", "concatenate"]:
            self.mode = mode.lower()
        else:
            raise ValueError("mode must be one of {'leave', 'combine', 'concatenate'}.")

        # get the reader classes
        self.rdr = getattr(io, reader)
        self.reader_kw = reader_kw

        # set an attribute if we want to skip file size checks
        if ignore_file_size_check:
            self.rdr._skip_file_size_check = True  # just needs to be set to anything

        self.resample_to_lowest = resample_to_lowest

        self.fill_gaps = fill_gaps
        self.fill_value = {} if fill_value is None else fill_value
        if gaps_error in ["raise", "warn", "ignore"]:
            self.gaps_error = gaps_error
        else:
            raise ValueError("gaps_error must be one of {'raise', 'warn', 'ignore'}.")

        self.require_all_keys = require_all_keys

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
        
        # drop any empty dictionaries - might occur due to some empty files
        res = [i for i in res if i]  # "if i" === "if i != {}"
        # check now that we have any data
        if not res:
            raise ValueError("No data found in any of the files.")

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
        
        # drop any empty dictionaries - might occur due to some empty files
        res = [i for i in res if i]  # "if i" === "if i != {}"
        # check now that we have any data
        if not res:
            raise ValueError("No data found in any of the files.")

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
                    if self.require_all_keys:
                        raise KeyError(
                            "To concatenate file contents, all files must have the same data streams."
                        )
                    else:
                        pass
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

    def handle_gaps_error(self, msg):
        if self.gaps_error == "raise":
            raise ValueError(msg)
        elif self.gaps_error == "warn":
            warn(msg)
        else:
            pass

    def check_handle_gaps(self, data):
        """
        Check for gaps, and fill if specified.

        Parameters
        ----------
        data : dict
            Dictionary of data-streams

        Returns
        -------
        res_dict : dict
            Dictionary of data with gaps filled, as specified.
        """
        if self.fill_gaps:
            # get a dictionary of streams that are the same size as the time array
            time = data.pop("time")
            stream_keys = [i for i in data if isinstance(data[i], ndarray)]
            stream_keys = [i for i in stream_keys if data[i].shape[0] == time.size]
            datastreams = {i: data.pop(i) for i in stream_keys}

            # get fs
            fs = data.pop("fs", None)
            if fs is None:
                fs = 1 / mean(diff(time[:5000]))
                warn(
                    f"Sampling frequency required calcluation, estimated to be {fs:.3f}. "
                    f"May be effected if there are data gaps in the first 5000 samples."
                )

            time_filled, data_filled = fill_data_gaps(
                time, fs, self.fill_value, **datastreams
            )

            if data:  # check if there are any keys left
                warn(
                    f"These keys could not be checked for gaps, or updated to new "
                    f"time series. If they are indices this may cause down-stream "
                    f"problems: {data.keys()}"
                )

            # put the non-filled data in the filled dictionary
            data_filled.update(data)
            data_filled["time"] = time_filled
            data_filled["fs"] = fs
        else:
            # check if any gaps and warn if there are
            fs = data.get(
                "fs", 1 / mean(diff(data["time"][:500]))
            )  # only get a few here
            time_deltas = diff(data["time"])
            if any(abs(time_deltas) > (1.5 / fs)):
                self.handle_gaps_error(
                    f"There are data gaps (max length: {max(time_deltas):.2f}s), "
                    f"which could potentially result in incorrect outputs from down-stream "
                    f"algorithms"
                )

            data_filled = data

        return data_filled

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
        if files is None:
            try:
                files = [kwargs.pop("file")]
            except KeyError:
                raise ValueError("No files provided to read.")

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

        # check for gaps, and fill if specified
        results = self.check_handle_gaps(results)

        # handle setting the file, either the first key from a dictionary or the
        # first index from a list
        results["file"] = files[i0]

        return results
