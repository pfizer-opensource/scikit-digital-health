"""
Stride creation and QC

Lukas Adamowicz
Copyright 2023 Pfizer Inc, All rights reserved
"""

from numpy import ones, zeros, nonzero, array, int_

from skdh.base import BaseProcess, handle_process_returns


class CreateStridesAndQc(BaseProcess):
    r"""
    Create strides from initial and final contacts. Perform a set of
    QC rules on the strides to eliminate strides that are not physiologically possible.

    Parameters
    ----------
    max_stride_time : {"default", callable, float}, optional
        The maximum stride time possible for a single stride. Either a callable
        with the input of the mean step time, or a float, which will be used as a
        static limit. Default ("default") is the function
        :math:`2.0 * mean\_step\_time + 1.0`.
    loading_factor : {"default", callable, float}, optional
        A factor that is multiplied by the `max_stride_time` to obtain values indicating
        the maximum initial double support and maximum stances times for strides to
        be deemed physiologically possible, and not discarded. Either a callable
        with the input of mean step time, or a float (between 0.0 and 1.0) indicating
        a static factor. Default ("default") is the function
        :math:`0.17 * mean\_step\_time + 0.05`.
    """

    def __init__(
        self,
        max_stride_time="default",
        loading_factor="default",
    ):
        super().__init__(
            max_stride_time=max_stride_time,
            loading_factor=loading_factor,
        )

        if max_stride_time == "default":
            self.max_stride_time_fn = lambda x: 2.0 * x + 1.0
        elif callable(max_stride_time):
            self.max_stride_time_fn = max_stride_time
        elif isinstance(max_stride_time, float):
            self.max_stride_time_fn = lambda x: max_stride_time
        else:
            raise ValueError(
                "`max_stride_time` is not a callable or a float, or 'default'."
            )

        if loading_factor == "default":
            self.loading_factor_fn = lambda x: 0.17 * x + 0.05
        elif callable(loading_factor):
            self.loading_factor_fn = loading_factor
        elif isinstance(loading_factor, float):
            if 0.0 < loading_factor < 1.0:
                self.loading_factor_fn = lambda x: loading_factor
            else:
                raise ValueError("`loading_factor` must be between 0.0 and 1.0")
        else:
            raise ValueError(
                "`loading_factor` is not a callable or a float, or 'default'."
            )

    @handle_process_returns(results_to_kwargs=True)
    def predict(
        self,
        time=None,
        initial_contacts=None,
        final_contacts=None,
        mean_step_freq=None,
        **kwargs,
    ):
        """
        predict(time, accel, initial_contacts, final_contacts, mean_step_freq)

        Parameters
        ----------
        time
        initial_contacts
        final_contacts
        mean_step_freq

        Returns
        -------
        results : dict
            Dictionary of the results, with the following items that can be used
            as inputs to downstream processing steps:

            - `qc_initial_contacts`: QC'ed initial contacts
            - `qc_final_contacts`: QC'ed final contacts, corresponding to `qc_initial_contacts`
            - `qc_final_contacts_oppfoot`: QC'ed final contacts of the opposite foot, corresponding
              to those in `qc_initial_contacts`
            - `forward_cycles`: Number of complete forward cycles that are available,
              corresponding to `qc_initial_contacts`.
        """
        # get the QC limits for this bout
        mean_step_time = 1 / mean_step_freq
        t_max_stride = self.max_stride_time_fn(mean_step_time)
        loading_factor = self.loading_factor_fn(mean_step_time)
        t_loading_forward = loading_factor * t_max_stride
        t_stance_forward = t_max_stride / 2 + t_loading_forward

        # get IC & FC times
        ic_times = time[initial_contacts]
        fc_times = time[final_contacts]

        # setup/tracking
        qc_ic, qc_fc, qc_fc_of = [], [], []
        fc_unused = ones(fc_times.size, dtype="bool")

        # check for any overlapping IC/FC events
        overlap = set(initial_contacts).intersection(set(final_contacts))
        for idx in overlap:
            fc_unused[final_contacts == idx] = False

        # iterate over available IC times
        for i, curr_ict in enumerate(ic_times[:-1]):
            # forward FC
            forward_idx = nonzero((fc_times > curr_ict) & fc_unused)[0]
            fc_forward = final_contacts[forward_idx]
            fc_forward_times = fc_times[forward_idx]

            if fc_forward.size == 0:
                continue

            # QC 0 :: can't have a second IC before the first FC
            if (ic_times[i + 1] - curr_ict) < (fc_forward_times[0] - curr_ict):
                continue

            # QC 1 :: must be a FC within the maximum initial double support (IDS) time
            # this would be the opposite foot FC
            if (fc_forward_times < (curr_ict + t_loading_forward)).sum() != 1:
                continue

            # QC 2 :: must be a second FC within the maximum step + IDS time
            # this would be the same foot FC
            if (fc_forward_times < (curr_ict + t_stance_forward)).sum() < 2:
                continue

            # create the stride
            qc_ic.append(initial_contacts[i])
            qc_fc.append(fc_forward[1])
            qc_fc_of.append(fc_forward[0])

            fc_unused[forward_idx[0]] = False

        qc_ic = array(qc_ic, dtype=int_)
        qc_fc = array(qc_fc, dtype=int_)
        qc_fc_of = array(qc_fc_of, dtype=int_)

        forward_cycles = zeros(qc_ic.size, dtype=int_)
        # QC 3 :: are there 2 forward cycles within the maximum stride time
        if forward_cycles.size > 2:
            forward_cycles[:-2] += (time[qc_ic[2:]] - time[qc_ic[:-2]]) < t_max_stride
        # is the next step continuous
        if forward_cycles.size > 1:
            forward_cycles[:-1] += qc_fc_of[1:] == qc_fc[:-1]

        res = {
            "qc_initial_contacts": qc_ic,
            "qc_final_contacts": qc_fc,
            "qc_final_contacts_oppfoot": qc_fc_of,
            "forward_cycles": forward_cycles,
        }

        return res
