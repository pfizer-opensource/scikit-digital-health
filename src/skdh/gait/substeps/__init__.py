r"""
.. _bout processing:

Gait Bout Processing (:mod:`skdh.gait.substeps`)
==================================================

.. currentmodule:: skdh.gait.substeps

.. autosummary::
    :toctree: generated/

    PreprocessGaitBout
    VerticalCwtGaitEvents
    ApCwtGaitEvents
    CreateStridesAndQc
    TurnDetection

New Bout Processing Steps
-----------------------------
To create a new bout processing step, simply create a new subclass of `skdh.BaseProcess`
and set it up for your processing needs.

Example code is below (from `CreateStridesAndQc`):

.. code-block:: python

    from numpy import ones, zeros, nonzero, array, int_

    from skdh.base import BaseProcess, handle_process_returns


    class CreateStridesAndQc(BaseProcess):
        '''
        Top docstring
        '''

        def __init__(
            self,
            max_stride_time=lambda x: 2.0 * x + 1.0,
            loading_factor=lambda x: 0.17 * x + 0.05,
        ):
            super().__init__(
                max_stride_time=max_stride_time,
                loading_factor=loading_factor,
            )

            # ... init code

        # return_to_kwargs=True indicates that the `res` dictionary will be
        # added to the inputs for any future processing steps
        @handle_process_returns(results_to_kwargs=True)
        def predict(
            self,
            time=None,
            initial_contacts=None,
            final_contacts=None,
            mean_step_freq=None,
            **kwargs,  # catches inputs required for future/previous processing steps
        ):
            '''
            predict docstring
            '''
            # processing

            # setup results from this function
            res = {
                "qc_initial_contacts": qc_ic,
                "qc_final_contacts": qc_fc,
                "qc_final_contacts_oppfoot": qc_fc_of,
                "forward_cycles": forward_cycles,
            }
    
            return res
"""

from skdh.gait.substeps.s1_preprocessing import PreprocessGaitBout
from skdh.gait.substeps.s2_contact_events import (
    VerticalCwtGaitEvents,
    ApCwtGaitEvents,
)
from skdh.gait.substeps.s3_stride_creation import CreateStridesAndQc
from skdh.gait.substeps.s4_turns import TurnDetection

__all__ = [
    "PreprocessGaitBout",
    "VerticalCwtGaitEvents",
    "ApCwtGaitEvents",
    "CreateStridesAndQc",
    "TurnDetection",
]
