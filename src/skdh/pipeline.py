"""
Pipeline class for stringing multiple features into 1 processing pipeline

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""

from operator import attrgetter
from importlib import import_module
from warnings import warn
import logging
from packaging import version
from copy import copy
from pathlib import Path

import yaml
from skdh.base import BaseProcess as Process


class NotAProcessError(Exception):
    pass


class ProcessNotFoundError(Exception):
    pass


class VersionError(Exception):
    pass


# update the YAML safe loader to handle Python tuples
class TupleSafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    TupleSafeLoader.construct_python_tuple,
)


def warn_or_raise(msg, err, err_raise):
    if err_raise:
        raise err(msg)
    else:
        warn(msg, UserWarning)


class Pipeline:
    """
    Pipeline class that can have multiple steps that are processed sequentially.
    Some of the output is passed between steps. Has the ability to save results
    from processing steps as local files, as well as return the results in a dictionary
    following the processing.

    Parameters
    ----------
    load_kwargs : {None, dict}, optional
        Dictionary of key-word arguments that will get directly passed to the
        `Pipeline.load()` function. If None, no pipeline will be loaded (default).
    flatten_results : bool, optional
        Flatten the results of the processing steps in the return dictionary. By
        default (False), results of a step will be stored under a key of the step's
        class name. If True, all results will be on the same level, and an exception
        will be raised if keys would be overwritten.

    Examples
    --------
    Load a pipeline, saved in a file, on instantiation. Also set it to raise an
    error instead of a warning if one of the processes cannot be loaded:

    >>> pipe = Pipeline(
    >>>     load_kwargs={"file": "example_pipeline.skdh", "process_raise": False})
    """

    def __str__(self):
        return "IMUAnalysisPipeline"

    def __repr__(self):
        ret = "IMUAnalysisPipeline[\n"
        for proc in self._steps:
            ret += f"\t{proc!r},\n"
        ret += "]"
        return ret

    def __init__(self, load_kwargs=None, flatten_results=False):
        self._steps = []
        self._save = []
        self._current = -1  # iteration tracking

        self.logger = logging.getLogger(__name__)

        self._min_vers = None

        # keeping track of duplicate processes
        self.duplicate_procs = {}

        if load_kwargs is not None:
            self.load(**load_kwargs)

        self.flatten_results = flatten_results

    def save(self, file):
        """
        Save the pipeline to a file for consistent pipeline generation.

        Parameters
        ----------
        file : {str, path-like}
            File path to save the pipeline structure to
        """
        # avoid circular import
        from skdh import __skdh_version__ as skdh_version

        pipe = {"Steps": [], "Version": skdh_version}
        for step in self._steps:  # actually look at steps
            package, module = step.__class__.__module__.split(".", 1)
            pipe["Steps"].append(
                {
                    "name": step._name,
                    "process": step._cls_name,
                    "package": package,
                    "module": module,
                    "parameters": step._kw,
                    "save_file": step.pipe_save_file,
                    "plot_file": step.pipe_plot_file,
                }
            )

        if Path(file).suffix != ".skdh":
            file += ".skdh"

        with open(file, "w") as f:
            yaml.dump(pipe, f)

    @staticmethod
    def _handle_load_input(yaml_str, file):
        """
        Handle different extensions loading

        Parameters
        ----------
        yaml_str : str
            YAML string of the pipeline. If provided, `file` is ignored. Supersedes
            the `json_str` parameter.
        file : str, Path-like
            File path to load

        Returns
        -------
        data : dict
            Loaded pipeline data.
        """
        valid_yaml_ext = [".skdh", ".yml", ".yaml"]

        if yaml_str is not None:
            return yaml.load(yaml_str, Loader=TupleSafeLoader)
        else:
            pfile = Path(file)
            if pfile.suffix not in valid_yaml_ext:
                warn(
                    f"File ({file}) does not have one of the expected suffixes: "
                    f"{valid_yaml_ext}",
                    UserWarning,
                )
            with open(file, "r") as f:
                data = yaml.load(f, Loader=TupleSafeLoader)
            return data

    def load(
        self,
        file=None,
        *,
        yaml_str=None,
        process_raise=False,
        noversion_raise=False,
        old_raise=False,
    ):
        """
        Load a previously saved pipeline from a file or YAML string.

        Parameters
        ----------
        file : {str, path-like}
            File path to load the pipeline structure from.
        yaml_str : str
            YAML string of the pipeline. If provided, `file` is ignored.
        process_raise : bool, optional
            Raise an error if a process in `file` or `yaml_str` cannot
            be added to the pipeline. Default is False, which issues a warning
            instead.
        noversion_raise : bool
            Raise an error if no version is provided in the input data. Default
            is False, which issues a warning instead.
        old_raise : bool
            Raise an error if the version used to create the pipeline is old enough
            to potentially cause compatibility/functionality errors. Default is False,
            which issues a warning instead.
        """
        import skdh

        min_vers = (
            skdh.__minimum_version__ if self._min_vers is None else self._min_vers
        )

        # import the data, handling possible input/file types
        data = self._handle_load_input(yaml_str, file)

        if "Steps" in data and "Version" in data:
            procs = data["Steps"]
            saved_version = data["Version"]
        else:
            warn_or_raise(
                "Pipeline created by an unknown version of skdh. Functionality not "
                "guaranteed.",
                VersionError,
                noversion_raise,
            )

            procs = data
            saved_version = "0.0.1"

        if version.parse(saved_version) < version.parse(min_vers):
            warn_or_raise(
                f"Pipeline was created by an older version of skdh ({saved_version}), "
                f"which may not be compatible with the current version "
                f"({skdh.__version__}). Functionality is not guaranteed.",
                VersionError,
                old_raise,
            )

        for proc in procs:
            if "process" in proc:
                name = proc.get("name", None)
                process_name = proc["process"]
                pkg = proc["package"]
                mod = proc["module"]
                params = proc["parameters"]
                save_file = proc["save_file"]
                plot_file = proc["plot_file"]
            else:
                warn(
                    "Save formats with class names as top-level keys in a list of "
                    "dictionaries will no longer be supported in a future release. "
                    "The new format can be used by loading and re-saving the pipeline file",
                    FutureWarning,
                )
                name = None
                process_name = list(proc.keys())[0]
                pkg = proc[process_name]["package"]
                mod = proc[process_name]["module"]
                params = proc[process_name]["parameters"]
                save_file = proc[process_name]["save_file"]
                plot_file = proc[process_name]["plot_file"]

            if pkg == "skdh":
                package = skdh
            else:
                package = import_module(pkg)

            try:
                process = attrgetter(f"{mod}.{process_name}")(package)
            except AttributeError:
                warn_or_raise(
                    f"Process ({pkg}.{mod}.{process_name}) not found. Not added to pipeline",
                    ProcessNotFoundError,
                    process_raise,
                )
                continue

            proc = process(**params)

            self.add(proc, name=name, save_file=save_file, plot_file=plot_file)

    def add(self, process, name=None, save_file=None, plot_file=None, make_copy=True):
        """
        Add a processing step to the pipeline

        Parameters
        ----------
        process : Process
            Process class that forms the step to be run
        name : str, optional
            Process name. Used to delineate multiple of the same process if
            required. Output results will be under this name. If None is provided,
            output results will be under the class name, with some mangling in the
            case of multiple of the same processes in the same pipeline.
        save_file : {None, str}, optional
            Optionally formattable path for the save file. If left/set to None,
            the results will not be saved anywhere.
        plot_file : {None, str}, optional
            Optionally formattable path for the output of plotting. If left/set
            to None, the plot will not be generated and saved.
        make_copy : bool, optional
            Create a shallow copy of `process` to add to the pipeline. This allows
            a single instance to be used in multiple pipelines while retaining custom
            save file names and other pipeline-specific attributes. Default is True.

        Notes
        -----
        Some of the avaible parameters for results saving and plotting are:

        - date : the current date, expressed as YYYYMMDD.
        - name : the name of the process doing the analysis.
        - file : the name of the input file passed to the pipeline.

        Note that if no file was passed in initially, then the `file` would be
        an empty string. However, even if the first step of the pipeline is
        not one that would use a `file` keyword, you can still specify it and
        it will be ignored for everything but this parameter.

        >>> p = Pipeline()
        >>> p.add(Gait(), save_file="{file}_gait_results.csv")
        >>> p.run(accel=accel, time=time)
        No file was passed in, so the resulting output file would be
        `_gait_results.csv`

        However if the `p.run` call is now:

        >>> p.run(accel=accel, time=time, file="example_file.txt")
        then the output would be `example_file_gait_results.csv`.

        Examples
        --------
        Add `Gait` and save the results to a fixed file name:

        >>> from skdh.gait import Gait
        >>> p = Pipeline()
        >>> p.add(Gait(), save_results="gait_results.csv")

        Add a binary file reader without saving the results and gait processing
        with a variable file name:

        >>> from skdh.io import ReadBin
        >>> p = Pipeline()
        >>> p.add(ReadBin(bases=0, periods=24), save_results=None)
        >>> p.add(Gait(), save_results="{date}_{name}_results.csv")

        If the date was, for example, May 18, 2021 then the results file would
        be `20210518_Gait_results.csv`.
        """
        if not isinstance(process, Process):
            raise NotAProcessError(
                f"process is not a subclass of {Process!r}, "
                "cannot be added to the pipeline"
            )
        if not make_copy:
            proc = process
        else:
            proc = copy(process)

        # attach the save bool and save_name to the process
        proc._in_pipeline = True
        proc.pipe_save_file = save_file
        proc.pipe_plot_file = plot_file

        # setup plotting
        proc._setup_plotting(plot_file)

        # point the step logging disabled to the pipeline disabled
        proc.logger.disabled = self.logger.disabled

        # setup the name
        if name is None:
            name = proc._cls_name

        # check if already in pipeline
        if name in self.duplicate_procs:
            self.duplicate_procs[name] += 1
            name = f"{name}_{self.duplicate_procs[name]}"
        else:
            self.duplicate_procs[name] = 0

        # set the process name
        proc._name = name

        self._steps += [proc]

    def __iter__(self):
        return self

    def __next__(self):
        self._current += 1
        if self._current < len(self._steps):
            return self._steps[self._current]

        self._current = -1  # reset for next run
        raise StopIteration

    def run(self, **kwargs):
        """
        Run through the pipeline, sequentially processing steps. Inputs must be
        provided as key-word arguments.

        Parameters
        ----------
        kwargs
            Any key-word arguments. Will get passed to the first step of the pipeline,
            and therefore they must contain at least what the first process is
            expecting.

        Returns
        -------
        results : dict
            Dictionary of the results of any steps of the pipeline that return results.
        """
        # set self._current to restart processing
        self._current = -1
        results = {}

        for proc in self:
            kwargs, step_result = proc.predict(**kwargs)
            if proc.pipe_save_file is not None:
                proc.save_results(
                    step_result if step_result is not None else kwargs,
                    proc.pipe_save_file,
                )
            if step_result is not None:
                if self.flatten_results:
                    if any(i in results for i in step_result):
                        raise IndexError(
                            "Results dictionary already contains values in the results, "
                            "cannot create a flat dictionary. Try setting `flatten_results=False`."
                        )
                    results.update(step_result)
                else:
                    results[proc._name] = step_result

        return results
