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

        self.logger = None
        self.log_filename = None

    def enable_logging(self, logger=None, filename='pipeline_log.log', name=None):
        """
        Enable logging during the execution of the pipeline. This will initialize loggers
        for steps already added, and any new steps that are added to the pipeline.

        Parameters
        ----------
        logger : logging.Logger, optional
            Custom logger for logging. If None (default), a logger will be created. See Notes for
            a more detailed description of the logger that will be created.
        filename : str, optional
            Filename/path for the logger. Default is 'pipeline_log.log'.
        name : str, optional
            Name for the logger. Default is None, which will use __name__ (skimu.pipeline)

        Notes
        -----
        If a logger is being created, it is created with the following attributes/parameters:

        - Name is set to __name__ (`name=None`) or `name`
        - Level is set to `INFO`
        - Handler is a `logging.FileHandler` with `filename` for the file
        """
        if logger is None:
            self.log_filename = filename  # save the file name
            self.logger = logging.getLogger(name if name is not None else __name__)
            self.logger.setLevel('INFO')  # set the level for the logger
            self.logger.addHandler(logging.FileHandler(filename=filename))
        else:
            if isinstance(logger, logging.Logger):
                self.logger = logger

        for step in self._steps:
            step.enable_logging(filename=filename)

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
                        'module': step.__class__.__module__.split('.', 1)[1],
                        'Parameters': step._kw,
                        'save_result': step.pipe_save,
                        'save_name': step.pipe_fname
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

        with open(file, 'r') as f:
            procs = json.load(f)

        for proc in procs:
            name = list(proc.keys())[0]
            mod = proc[name]["module"]
            params = proc[name]["Parameters"]
            save_result = proc[name]["save_result"]
            save_name = proc[name]["save_name"]

            try:
                process = attrgetter(f"{mod}.{name}")(skimu)
            except AttributeError:
                warn(
                    f"Process (skimu.{mod}.{name}) not found. Not being added to pipeline",
                    UserWarning
                )
                continue

            self.add(
                process(**params),
                save_results=save_result,
                save_name=save_name
            )

    def add(self, process, save_results=False, save_name="{date}_{name}_results.cv",
            logging_name=None):
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
        logging_name : str, optional
            Name for the logger for the process, if `enable_logging` has been called. Default is
            None, which will use the __name__ attribute for the process.
        """
        if not isinstance(process, Process):
            raise NotAProcessError("process is not a subclass of _BaseProcess, "
                                   "cannot be added to the pipeline")

        self._steps += [process]
        # attach the save bool and save_name to the process
        self._steps[-1].pipe_save = save_results
        self._steps[-1].pipe_fname = save_name

        if self.logger is not None:
            self._steps[-1].enable_logging(name=logging_name, filename=self.log_filename)

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
            kwargs, step_result = proc._predict(**kwargs)
            if proc.pipe_save:
                proc.save_results(step_result if proc._return_result else kwargs, proc.pipe_fname)
            if proc._return_result:
                results[proc._name] = step_result

        return results
