"""
Cutpoint definitions

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from skdh.activity import metrics


def get_available_cutpoints(name=None):
    """
    Print the available cutpoints for activity level segmentation, or the
    thresholds for a specific set of cutpoints.

    Parameters
    ----------
    name : {None, str}, optional
        The name of the cupoint values to print. If None, will print all
        the available cutpoint options.
    """
    if name is None:
        for k in _base_cutpoints:
            print(k)
    else:
        cuts = _base_cutpoints[name]

        print(f"{name}\n{'-' * 15}")
        print(f"Metric: {cuts['metric']}")

        for level in ["sedentary", "light", "moderate", "vigorous"]:
            lthresh, uthresh = get_level_thresholds(level, cuts)

            print(f"{level} range [g]: {lthresh:0.3f} -> {uthresh:0.3f}")


def get_level_thresholds(level, cutpoints):
    if level.lower() in ["sed", "sedentary"]:
        return -1e5, cutpoints["sedentary"]
    elif level.lower() == "light":
        return cutpoints["sedentary"], cutpoints["light"]
    elif level.lower() in ["mod", "moderate"]:
        return cutpoints["light"], cutpoints["moderate"]
    elif level.lower() in ["vig", "vigorous"]:
        return cutpoints["moderate"], 1e5
    elif level.lower() == "mvpa":
        return cutpoints["light"], 1e5
    elif level.lower() == "slpa":  # sedentary-light phys. act.
        return -1e5, cutpoints["light"]
    else:
        raise ValueError(f"Activity level label [{level}] not recognized.")


def get_metric(name):
    return getattr(metrics, name)


# ==========================================================
# Activity cutpoints
_base_cutpoints = {}

_base_cutpoints["esliger_lwrist_adult"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": True},
    "sedentary": 217 / 80 / 60,  # paper at 80hz, summed for each minute long window
    "light": 644 / 80 / 60,
    "moderate": 1810 / 80 / 60,
}

_base_cutpoints["esliger_rwirst_adult"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": True},
    "sedentary": 386 / 80 / 60,  # paper at 80hz, summed for each 1min window
    "light": 439 / 80 / 60,
    "moderate": 2098 / 80 / 60,
}

_base_cutpoints["esliger_lumbar_adult"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": True},
    "sedentary": 77 / 80 / 60,  # paper at 80hz, summed for each 1min window
    "light": 219 / 80 / 60,
    "moderate": 2056 / 80 / 60,
}

_base_cutpoints["schaefer_ndomwrist_child6-11"] = {
    "metric": "metric_bfen",
    "kwargs": {"low_cutoff": 0.2, "high_cutoff": 15, "trim_zero": False},
    "sedentary": 0.190,
    "light": 0.314,
    "moderate": 0.998,
}

_base_cutpoints["phillips_rwrist_child8-14"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": True},
    "sedentary": 6 / 80,  # paper at 80hz, summed for each 1s window
    "light": 21 / 80,
    "moderate": 56 / 80,
}

_base_cutpoints["phillips_lwrist_child8-14"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": True},
    "sedentary": 7 / 80,
    "light": 19 / 80,
    "moderate": 60 / 80,
}

_base_cutpoints["phillips_hip_child8-14"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": True},
    "sedentary": 3 / 80,
    "light": 16 / 80,
    "moderate": 51 / 80,
}

_base_cutpoints["vaha-ypya_hip_adult"] = {
    "metric": "metric_mad",
    "kwargs": {},
    "light": 0.091,  # originally presented in mg
    "moderate": 0.414,
}

_base_cutpoints["hildebrand_hip_adult_actigraph"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0474,
    "light": 0.0691,
    "moderate": 0.2587,
}

_base_cutpoints["hildebrand_hip_adult_geneactv"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0469,
    "light": 0.0687,
    "moderate": 0.2668,
}

_base_cutpoints["hildebrand_wrist_adult_actigraph"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0448,
    "light": 0.1006,
    "moderate": 0.4288,
}

_base_cutpoints["hildebrand_wrist_adult_geneactiv"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0458,
    "light": 0.0932,
    "moderate": 0.4183,
}

_base_cutpoints["hildebrand_hip_child7-11_actigraph"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0633,
    "light": 0.1426,
    "moderate": 0.4646,
}

_base_cutpoints["hildebrand_hip_child7-11_geneactiv"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0641,
    "light": 0.1528,
    "moderate": 0.5143,
}

_base_cutpoints["hildebrand_wrist_child7-11_actigraph"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0356,
    "light": 0.2014,
    "moderate": 0.707,
}

_base_cutpoints["hildebrand_wrist_child7-11_geneactiv"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.0563,
    "light": 0.1916,
    "moderate": 0.6958,
}

_base_cutpoints["migueles_wrist_adult"] = {
    "metric": "metric_enmo",
    "kwargs": {"take_abs": False, "trim_zero": True},
    "sedentary": 0.050,
    "light": 0.110,
    "moderate": 0.440,
}
