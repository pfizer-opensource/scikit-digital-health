"""
Pipeline class for stringing multiple features into 1 processing pipeline

Lukas Adamowicz
Pfizer DMTI 2020
"""
from skimu.base import _BaseProcess


class NotAProcessError(Exception):
    pass


class Pipeline:
    def __str__(self):
        return "IMUAnalysisPipeline"

    def __repr__(self):
        ret = "["
        for proc in self._steps:
            ret += f'\t{proc!r},\n'
        ret = ret[:-2] + ']'
        return ret

    def __init__(self):
        """
        Pipeline class that can have multiple steps that are processed sequentially
        """
        self._steps = []

    def add(self, process):
        """
        Add a processing step to the pipeline

        Parameters
        ----------
        process : Process
            Process class that forms the step to be run
        """
        if not isinstance(process, _BaseProcess):
            raise NotAProcessError("process is not a subclass of _BaseProcess, cannot be added "
                                   "to the pipeline")

        self._steps += [process]

    def run(self, *args, **kwargs):
        """
        Run through the pipeline, sequentially processing steps

        Parameters
        ----------
        args
            Any positional arguments. Will get passed to the first step of the pipeline
        kwargs
            Any key-word arguments. Will get passed to the first step of the pipeline

        Returns
        -------
        results : dict
            Dictionary of the results of any steps of the pipeline that return results
        """
        results = {}

        # treat the first step specially due to args, kwargs input
        inout, step_result = self._steps[0]._predict(*args, **kwargs)
        if self._steps[0]._return_result:
            results[self._steps[0]._proc_name] = step_result

        # iterate over the rest of the processes
        for process in self._steps[1:]:
            inout, step_result = process._predict(**inout)
            if process._return_result:
                results[process._proc_name] = step_result

        return results
