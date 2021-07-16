"""
Pipeline class for stringing multiple features into 1 processing pipeline

Lukas Adamowicz
Pfizer DMTI 2020
"""
import json
from operator import attrgetter
from importlib import import_module
from warnings import warn
import logging

from skdh.base import BaseProcess as Process


class NotAProcessError(Exception):
    pass


class ProcessNotFoundError(Exception):
    pass


class Pipeline:
    """
    Pipeline class that can have multiple steps that are processed sequentially. Some of the output
    is passed between steps. Has the ability to save results from processing steps as local files,
    as well as return the results in a dictionary following the processing.

    Parameters
    ----------
    file : {None, str}, optional
        File path to load a pipeline from. Default is None, in which case no
        file will be loaded.
    """

    def __str__(self):
        return "IMUAnalysisPipeline"

    def __repr__(self):
        ret = "IMUAnalysisPipeline[\n"
        for proc in self._steps:
            ret += f"\t{proc!r},\n"
        ret += "]"
        return ret

    def __init__(self, file=None):
        self._steps = []
        self._save = []
        self._current = -1  # iteration tracking

        self.logger = logging.getLogger(__name__)

        if file is not None:
            self.load(file=file)

    def save(self, file):
        """
        Save the pipeline to a file for consistent pipeline generation.

        Parameters
        ----------
        file : {str, path-like}
            File path to save the pipeline structure to
        """
        pipe = []
        for step in self._steps:  # actually look at steps
            package, module = step.__class__.__module__.split(".", 1)
            pipe.append(
                {
                    step._name: {
                        "package": package,
                        "module": module,
                        "parameters": step._kw,
                        "save_file": step.pipe_save_file,
                        "plot_file": step.pipe_plot_file,
                    }
                }
            )

        with open(file, "w") as f:
            json.dump(pipe, f)

    def load(self, file):
        """
        Load a previously saved pipeline from a file

        Parameters
        ----------
        file : {str, path-like}
            File path to load the pipeline structure from
        """
        import skdh

        with open(file, "r") as f:
            procs = json.load(f)

        for proc in procs:
            name = list(proc.keys())[0]
            pkg = proc[name]["package"]
            mod = proc[name]["module"]
            params = proc[name]["parameters"]
            save_file = proc[name]["save_file"]
            plot_file = proc[name]["plot_file"]

            if pkg == "skdh":
                package = skdh
            else:
                package = import_module(pkg)

            try:
                process = attrgetter(f"{mod}.{name}")(package)
            except AttributeError:
                warn(
                    f"Process ({pkg}.{mod}.{name}) not found. Not added to pipeline",
                    UserWarning,
                )
                continue

            proc = process(**params)

            self.add(proc, save_file=save_file, plot_file=plot_file)

    def add(self, process, save_file=None, plot_file=None):
        """
        Add a processing step to the pipeline

        Parameters
        ----------
        process : Process
            Process class that forms the step to be run
        save_file : {None, str}, optional
            Optionally formattable path for the save file. If left/set to None,
            the results will not be saved anywhere.
        plot_file : {None, str}, optional
            Optionally formattable path for the output of plotting. If left/set
            to None, the plot will not be generated and saved.

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

        >>> from skdh.read import ReadBin
        >>> p = Pipeline()
        >>> p.add(ReadBin(bases=0, periods=24), save_results=None)
        >>> p.add(Gait(), save_results="{date}_{name}_results.csv")

        If the date was, for example, May 18, 2021 then the results file would
        be `20210518_Gait_results.csv`.
        """
        if not isinstance(process, Process):
            raise NotAProcessError(
                "process is not a subclass of _BaseProcess, "
                "cannot be added to the pipeline"
            )

        # attach the save bool and save_name to the process
        process._in_pipeline = True
        process.pipe_save_file = save_file
        process.pipe_plot_file = plot_file

        # setup plotting
        process._setup_plotting(plot_file)

        # point the step logging disabled to the pipeline disabled
        process.logger.disabled = self.logger.disabled

        self._steps += [process]

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
        Run through the pipeline, sequentially processing steps. Inputs must be provided as
        key-word arguments

        Parameters
        ----------
        kwargs
            Any key-word arguments. Will get passed to the first step of the pipeline, and
            therefore they must contain at least what the first process is expecting.

        Returns
        -------
        results : dict
            Dictionary of the results of any steps of the pipeline that return results
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
                results[proc._name] = step_result

        return results
