"""
Utility functions for the Activity module

Lukas Adamowicz
Copyright (c) 2022. Pfizer Inc. All rights reserved.
"""

from warnings import warn

from skdh.activity.cutpoints import _base_cutpoints


def handle_cutpoints(cutpoints):
    """
    Handle multiple input types for the cutpoints argument

    Parameters
    ----------
    cutpoints : {str, dict, None}
        Cutpoints definition. Either a string pointing to an in-built
        list of cutpoints, or a dictionary of cutpoints to use.

    Returns
    -------
    cp : dict
        Dictionary of cutpoints
    """
    c1 = cutpoints is None
    c2 = cutpoints not in _base_cutpoints if isinstance(cutpoints, str) else False
    if c1 or c2:
        warn(
            f"Specified cutpoints not found, or cutpoints undefined. Using `migueles_wrist_adult`."
        )
        cp = _base_cutpoints["migueles_wrist_adult"]
    elif isinstance(cutpoints, str):
        cp = _base_cutpoints.get(cutpoints)
    elif isinstance(cutpoints, dict):
        # check that it has the appropriate entries
        rq_keys = ["metric", "kwargs", "sedentary", "light", "moderate"]
        missing_keys = [i for i in rq_keys if i not in cutpoints]
        if len(missing_keys) != 0:
            raise ValueError(f"User defined cutpoints missing keys {missing_keys}")

        cp = cutpoints
    else:
        raise ValueError(f"Cutpoints type {type(cutpoints)} not understood.")

    return cp
