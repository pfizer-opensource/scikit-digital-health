"""
Pipeline class for stringing multiple features into 1 processing pipeline

Lukas Adamowicz
Pfizer DMTI 2020
"""
import json
from operator import attrgetter
from warnings import warn
import logging

from skimu.base import _BaseProcess as Process


class NotAProcessError(Exception):
    pass


class ProcessNotFoundError(Exception):
    pass


class Pipeline:
    """
    Pipeline class that can have multiple steps that are processed sequentially. Some of the output
    is passed between steps. Has the ability to save results from processing steps as local files,
    as well as return the results in a dictionary following the processing.
    """
    def __str__(self):
        return "IMUAnalysisPipeline"

    def __repr__(self):
        ret = "["
        for proc in self._steps:
            ret += f'\t{proc!r},\n'
        ret = ret[:-2] + ']'
        return ret

    def __init__(self):
        self._steps = []
        self._save = []
        self._current = -1  # iteration tracking

        self.logger = logging.getLogger(__name__)

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
            pipe.append(
                {
                    step._name: {
                        "module": step.__class__.__module__.split('.', 1)[1],
                        "Parameters": step._kw,
                        "save_result": step.pipe_save,
                        "save_name": step.pipe_fname,
                        "plot_save_name": step.plot_fname
                    }
                }
            )

        with open(file, 'w') as f:
            json.dump(pipe, f)

    def load(self, file):
        """
        Load a previously saved pipeline from a file

        Parameters
        ----------
        file : {str, path-like}
            File path to load the pipeline structure from
        """
        import skimu

        with open(file, "r") as f:
            procs = json.load(f)

        for proc in procs:
            name = list(proc.keys())[0]
            mod = proc[name]["module"]
            params = proc[name]["Parameters"]
            save_result = proc[name]["save_result"]
            save_name = proc[name]["save_name"]
            plot_fname = proc[name]["plot_save_name"]

            try:
                process = attrgetter(f"{mod}.{name}")(skimu)
            except AttributeError:
                warn(
                    f"Process (skimu.{mod}.{name}) not found. Not being added to pipeline",
                    UserWarning
                )
                continue

            proc = process(**params)
            if plot_fname is not None:
                proc._setup_plotting(plot_fname)

            self.add(
                proc,
                save_results=save_result,
                save_name=save_name
            )

    def add(self, process, save_results=False, save_name="{date}_{name}_results.cv"):
        """
        Add a processing step to the pipeline

        Parameters
        ----------
        process : Process
            Process class that forms the step to be run
        save_results : bool
            Whether or not to save the results of the process as a csv. Default is False.
        save_name : str
            Optionally formattable path for the save file. Ignored if `save_results` is False.
            For options for the formatting, see any of the processes (e.g. :class:`Gait`,
            :class:`Sit2Stand`). Default is "{date}_{name}_results.csv
        """
        if not isinstance(process, Process):
            raise NotAProcessError("process is not a subclass of _BaseProcess, "
                                   "cannot be added to the pipeline")

        self._steps += [process]
        # attach the save bool and save_name to the process
        self._steps[-1]._in_pipeline = True
        self._steps[-1].pipe_save = save_results
        self._steps[-1].pipe_fname = save_name

        # point the step logging disabled to the pipeline disabled
        self._steps[-1].logger.disabled = self.logger.disabled

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
            if proc.pipe_save:
                proc.save_results(
                    step_result if step_result is not None else kwargs,
                    proc.pipe_fname
                )
            if step_result is not None:
                results[proc._name] = step_result

        return results
