from __future__ import annotations

import glob
import os.path
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from loguru import logger
from pandas._typing import FilePath

from crystalgraph import LQG, UQG
from crystalgraph.utils import json_dump
from crystalgraph.utils import json_load

"""
check pairwise isomorphism
"""


def get_qg_dicts(lqg_json_folder: FilePath):
    lqg_dict = dict()
    uqg_dict = dict()
    for jf in sorted(glob.glob(f"{lqg_json_folder}/*.json")):
        d = json_load(jf)
        lqg = LQG.from_dict(d)
        assert set(lqg.symbols.values()) == {'Si1 O4'}
        lqg_id = os.path.basename(jf)
        uqg = lqg.to_uqg()
        lqg_dict[lqg_id] = lqg
        uqg_dict[lqg_id] = uqg
    return lqg_dict, uqg_dict


def check_pairwise_isomorphism(qg_dict: dict[str, LQG | UQG], output: FilePath):
    eq_table = defaultdict(dict)
    n_eq = 0
    qg_dict_items = list(qg_dict.items())
    for i, (qg_id1, qg1) in tqdm.tqdm(list(enumerate(qg_dict_items))):
        for j, (qg_id2, qg2) in enumerate(qg_dict_items):
            if i > j: continue
            if i == j or qg1 == qg2:
                eq_table[qg_id1][qg_id2] = True
                eq_table[qg_id2][qg_id1] = True
                if i != j:
                    n_eq += 1
                    logger.warning(f"eq found! so far: {n_eq}")
            else:
                eq_table[qg_id1][qg_id2] = False
                eq_table[qg_id2][qg_id1] = False
    json_dump(eq_table, output)
    return eq_table


def ax_matrix(table_json: FilePath, ax: plt.Axes):
    df = pd.DataFrame(json_load(table_json))
    mat = df.values.astype(dtype=int)
    n_eq = int((df.sum(0).sum() - len(df)) / 2)
    ax.imshow(mat, cmap="Spectral", origin="lower")
    ax.set_title(f"# of unique isomorphic pairs: {n_eq}")


if __name__ == '__main__':
    l_dict, u_dict = get_qg_dicts("lqg_bu_json")
    check_pairwise_isomorphism(l_dict, "eq_table_lqg_bu.json")
    check_pairwise_isomorphism(u_dict, "eq_table_uqg_bu.json")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))
    ax_matrix("eq_table_uqg_bu.json", ax1)
    ax_matrix("eq_table_lqg_bu.json", ax2)
    fig.savefig("iso_check.png")
