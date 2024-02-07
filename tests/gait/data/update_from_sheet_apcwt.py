import numpy as np
import pandas as pd

keys = [
    "Bout N",
    "Bout Steps",
    "Gait Cycles",
    "m1 delta h",
    "m2 delta h",
    "m2 delta h prime",
    "step time",
    "stride time",
    "stance time",
    "swing time",
    "initial double support",
    "terminal double support",
    "double support",
    "single support",
    "step length",
    "stride length",
    "gait speed",
    "cadence",
]


if __name__ == "__main__":
    df = pd.read_excel("manual_gait_results_apcwt.xlsx")
    df_turn = pd.read_excel("manual_gait_turn_results_apcwt.xlsx")

    # list of values requiring only 1 forward cycle
    offset_1 = [
        "step time",
        "terminal double support",
        "double support",
        "single support",
        "step length",
        "cadence",
    ]
    # 2 forward cycles required
    offset_2 = [
        "stride time",
        "swing time",
        "stride length",
        "gait speed",
    ]

    # set these values to nan where appropriate
    for k in offset_1:
        df.loc[df["forward cycles"] < 1, k] = np.nan
        # df_turn.loc[df_turn["forward cycles"] < 1, k] = np.nan

    for k in offset_2:
        df.loc[df["forward cycles"] < 2, k] = np.nan
        # df_turn.loc[df_turn["forward cycles"] < 2, k] = np.nan

    # get keys to update results for
    keys_turn = keys + ["Turn"]

    tmp = {k: df.loc[:, k].values for k in keys}
    tmp_turn = {k: df_turn.loc[:, k].values for k in keys_turn}

    np.savez("gait_results_apcwt.npz", **tmp)
    np.savez("gait_results2_apcwt.npz", **tmp_turn)
