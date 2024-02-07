import numpy as np
import pandas as pd

keys = [
    "Bout N",
    "Bout Steps",
    "Gait Cycles",
    "m1 delta h",
    "step time",
    "step time asymmetry",
    "stride time",
    "stride time asymmetry",
    "stance time",
    "stance time asymmetry",
    "swing time",
    "swing time asymmetry",
    "initial double support",
    "initial double support asymmetry",
    "terminal double support",
    "terminal double support asymmetry",
    "double support",
    "double support asymmetry",
    "single support",
    "single support asymmetry",
    "step length m1",
    "step length m1 asymmetry",
    "stride length m1",
    "stride length m1 asymmetry",
    "gait speed m1",
    "gait speed m1 asymmetry",
    "cadence",
]


if __name__ == "__main__":
    df = pd.read_excel("manual_gait_results_vcwt.xlsx")
    df_turn = pd.read_excel("manual_gait_turn_results_vcwt.xlsx")

    # list of values requiring only 1 forward cycle
    offset_1 = [
        "step time",
        "terminal double support",
        "double support",
        "single support",
        "step length m1",
        "cadence",
        "stance time asymmetry",
        "initial double support asymmetry",
    ]
    # 2 forward cycles required
    offset_2 = [
        "stride time",
        "swing time",
        "stride length m1",
        "gait speed m1",
        "step time asymmetry",
        "terminal double support asymmetry",
        "double support asymmetry",
        "single support asymmetry",
        "step length m1 asymmetry",
    ]

    # 3 forward cycles required
    offset_3 = [
        "stride time asymmetry",
        "swing time asymmetry",
        "stride length m1 asymmetry",
        "gait speed m1 asymmetry",
    ]

    # set these values to nan where appropriate
    for k in offset_1:
        df.loc[df["forward cycles"] < 1, k] = np.nan
        df_turn.loc[df_turn["forward cycles"] < 1, k] = np.nan

    for k in offset_2:
        df.loc[df["forward cycles"] < 2, k] = np.nan
        df_turn.loc[df_turn["forward cycles"] < 2, k] = np.nan

    for k in offset_3:
        df.loc[df["forward cycles"] < 3, k] = np.nan
        df_turn.loc[df_turn["forward cycles"] < 3, k] = np.nan

    # get keys to update results for
    keys_turn = keys + ["Turn"]

    # old data file
    # old = np.load("gait_results_vcwt.npz")
    # old_turn = np.load("gait_results2_vcwt.npz")

    tmp = {k: df.loc[:, k].values for k in keys}
    # tmp.update(
    #     {k: np.full(tmp["Bout N"].size, np.nan) for k in old.files if k not in tmp}
    # )

    tmp_turn = {k: df_turn.loc[:, k].values for k in keys_turn}
    # tmp_turn.update(
    #     {
    #         k: np.full(tmp_turn["Bout N"].size, np.nan)
    #         for k in old_turn.files
    #         if k not in tmp_turn
    #     }
    # )

    np.savez("gait_results_vcwt.npz", **tmp)
    np.savez("gait_results2_vcwt.npz", **tmp_turn)
