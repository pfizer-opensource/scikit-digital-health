import numpy as np
import pandas as pd


if __name__ == "__main__":
    df = pd.read_excel("manual_gait_results.xlsx")
    df_turn = pd.read_excel("manual_gait_turn_results.xlsx")

    # list of values requiring only 1 forward cycle
    offset_1 = [
        "PARAM:step time",
        "PARAM:terminal double support",
        "PARAM:double support",
        "PARAM:single support",
        "PARAM:step length",
        "PARAM:cadence",
        "PARAM:stance time asymmetry",
        "PARAM:initial double support asymmetry",
    ]
    # 2 forward cycles required
    offset_2 = [
        "PARAM:stride time",
        "PARAM:swing time",
        "PARAM:stride length",
        "PARAM:gait speed",
        "PARAM:step time asymmetry",
        "PARAM:terminal double support asymmetry",
        "PARAM:double support asymmetry",
        "PARAM:single support asymmetry",
        "PARAM:step length asymmetry",
    ]

    # 3 forward cycles required
    offset_3 = [
        "PARAM:stride time asymmetry",
        "PARAM:swing time asymmetry",
        "PARAM:stride length asymmetry",
        "PARAM:gait speed asymmetry",
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
    keys = [i for i in df.columns if "PARAM" in i]
    keys = ["Bout N", "Bout Steps", "Gait Cycles", "delta h"] + keys

    keys_turn = [i for i in df_turn.columns if "PARAM" in i]
    keys_turn = ["Bout N", "Bout Steps", "Gait Cycles", "delta h", "Turn"] + keys_turn

    # old data file
    old = np.load("gait_results.npz")
    old_turn = np.load("gait_results2.npz")

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

    np.savez("gait_results.npz", **tmp)
    np.savez("gait_results2.npz", **tmp_turn)
