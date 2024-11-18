import glob
import os.path
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from loguru import logger
from pymatgen.core.structure import Structure
from tqdm import tqdm

from crystalgraph import LQG
from crystalgraph.features import LqgFeatureSet
from crystalgraph.utils import json_dump, FilePath, tqdm_joblib, json_load


def _parse_res(file: FilePath) -> dict:
    """
    diameters of, respectively,
    the largest included sphere,
    the largest free sphere,
    and the largest included sphere along free sphere path
    """
    with open(file, "r") as f:
        values = f.readline().split()[1:]
    keys = [
        "largest_included_sphere", "largest_free_sphere",
        "largest_included_sphere_along_free_sphere_path",
    ]
    return dict(zip(keys, values))


def _parse_chan(file: FilePath) -> dict:
    with open(file, "r") as f:
        first_line = f.readline().strip()
        items = first_line.split()[1:]
    total = int(items[0])
    if total == 0:
        values = [0, 0, 0]
    else:
        dims = [int(dim) for dim in items[-total:]]
        dim_counter = Counter(dims)
        values = [dim_counter[d] for d in [1, 2, 3]]
    values.append(sum(values))
    keys = ["num_1d_chan", "num_2d_chan", "num_3d_chan", "num_chan"]
    return dict(zip(keys, values))


def _parse_sa(file: FilePath) -> dict:
    # TODO add chan sa and pocket info
    with open(file, "r") as f:
        first_line = f.readline().strip()
    items = first_line.split()[2:]
    data = dict()
    for i in range(len(items)):
        if i % 2 == 0:
            data[items[i]] = float(items[i + 1])
    sa_data = {k: data[k] for k in ["Unitcell_volume:", "Density:", "ASA_A^2:", "NASA_A^2:", ]}
    return sa_data


def get_zpp_record(zeo_id: str, zpp_results_folder: FilePath):
    chan_file = f"{zpp_results_folder}/chan/{zeo_id}.chan"
    res_file = f"{zpp_results_folder}/res/{zeo_id}.res"
    sa_file = f"{zpp_results_folder}/sa/{zeo_id}.sa"
    if any(not os.path.isfile(f) for f in (chan_file, res_file, sa_file)):
        logger.critical(f"not all zpp results are found, record not created: zeo_id='{zeo_id}'")
        return
    data = dict()
    data["zeo_id"] = zeo_id
    data.update(_parse_res(res_file))
    data.update(_parse_sa(sa_file))
    data.update(_parse_chan(chan_file))
    return data


def get_zpp_df_parallel(
        zpp_results_folder: FilePath,
        zeo_ids: list[str],
        n_jobs=4
):
    assert os.path.isdir(f"{zpp_results_folder}/chan")
    assert os.path.isdir(f"{zpp_results_folder}/sa")
    assert os.path.isdir(f"{zpp_results_folder}/res")
    with joblib_progress("Prepare zpp dataset...", total=len(zeo_ids)):
        records = Parallel(n_jobs=n_jobs)(
            delayed(get_zpp_record)(zeo_id, zpp_results_folder) for zeo_id in zeo_ids)
    logger.info(f"organize results for # of zeolites: {len(zeo_ids)}")
    logger.info(f"# records obtained: {len(records)}")

    df = pd.DataFrame.from_records(records)
    return df


def export_lqg(structure_file: FilePath, pbu_lqg_folder: FilePath, lqg_folder: FilePath):
    try:
        zeo = Structure.from_file(structure_file)
        zeo_id = Path(structure_file).stem

        lqg = LQG.from_structure(zeo)
        d = lqg.as_dict()
        json_dump(d, f"{lqg_folder}/{zeo_id}.json")

        lqg_bu = lqg.bu_contraction()
        d = lqg_bu.as_dict()
        json_dump(d, f"{pbu_lqg_folder}/{zeo_id}.json")
        return zeo_id
    except Exception as e:
        logger.error(e.__str__())


def export_lqg_parallel(cssr_folder: FilePath, pbu_lqg_folder: FilePath, lqg_folder: FilePath, random_sample=False,
                        k=None, n_jobs=1):
    structure_files = sorted(glob.glob(f"{cssr_folder}/*.cssr"))
    if random_sample:
        assert k is not None
        random.seed(42)
        structure_files = random.sample(structure_files, k)
    elif k is not None:
        structure_files = structure_files[:k]

    if n_jobs == 1:
        zeo_ids = []
        for jf in tqdm(structure_files):
            try:
                zeo_id = export_lqg(jf, pbu_lqg_folder, lqg_folder)
                zeo_ids.append(zeo_id)
            except Exception as e:
                logger.warning(e.__str__())
                continue
    else:
        with tqdm_joblib(tqdm(desc="parallel export lqg", total=len(structure_files))) as progress_bar:
            zeo_ids = Parallel(n_jobs=n_jobs)(
                delayed(export_lqg)(structure_files[i], pbu_lqg_folder, lqg_folder) for i in
                range(len(structure_files)))
    return zeo_ids


def export_lqg_feature_record(qg: LQG, lqg_feature_folder: FilePath, identifier: str):
    lfs = LqgFeatureSet.from_lqg(qg)
    r = lfs.as_dict()
    json_dump(r, f"{lqg_feature_folder}/{identifier}.json")
    return r


def export_lqg_feature_record_parallel(lqg_feature_folder: FilePath, lqgs: list[LQG], prop_records: list[dict],
                                       n_jobs=1):
    if n_jobs == 1:
        feature_records = []
        for qg, prop in tqdm(zip(lqgs, prop_records)):
            r = export_lqg_feature_record(qg, lqg_feature_folder, prop["zeo_id"])
            feature_records.append(r)
    else:
        with tqdm_joblib(tqdm(desc="calculate lfs...", total=len(lqgs))) as progress_bar:
            feature_records = Parallel(n_jobs=n_jobs)(
                delayed(export_lqg_feature_record)(qg, prop["zeo_id"]) for qg, prop in zip(lqgs, prop_records))
    return feature_records


def load_lqg_parallel(lqg_json_files: list[FilePath], n_jobs=1):
    def load_lqg(json_file: FilePath) -> LQG:
        d = json_load(json_file)
        return LQG.from_dict(d)

    if n_jobs == 1:
        graphs = []
        for jf in tqdm(lqg_json_files):
            graphs.append(load_lqg(jf))
    else:
        with tqdm_joblib(tqdm(desc="load lqgs...", total=len(lqg_json_files))) as progress_bar:
            graphs = Parallel(n_jobs=5)(delayed(load_lqg)(jf) for jf in lqg_json_files)
    return graphs


def main(
        cssr_folder: FilePath = "cssr",
        lqg_folder: FilePath = "LQG",
        pbu_lqg_folder: FilePath = "PBU_LQG",
        lqg_feature_folder: FilePath = "LQG_feature",
):
    assert os.path.isdir(f"{cssr_folder}"), "CSSR folder not found!"
    Path(lqg_folder).mkdir(exist_ok=True)
    Path(pbu_lqg_folder).mkdir(exist_ok=True)
    Path(lqg_feature_folder).mkdir(exist_ok=True)

    # step 1: export LQG
    zeo_ids = export_lqg_parallel(cssr_folder, pbu_lqg_folder, lqg_folder, n_jobs=1)

    # step 2: run and export zpp results, these are model targets
    df_zpp = get_zpp_df_parallel("./", zeo_ids, n_jobs=4)
    df_zpp.to_csv(f"data_zpp.csv", index=False)
    zeo_ids = df_zpp['zeo_id'].tolist()

    # step 3: get silicate lqgs
    lqg_json_files = [f"{lqg_folder}/{lqg_id}.json" for lqg_id in zeo_ids]
    lqgs = load_lqg_parallel(lqg_json_files, n_jobs=4)
    prop_records = []
    silicate_lqgs = []
    for g, rid, r in tqdm(zip(lqgs, zeo_ids, df_zpp.to_dict(orient="records"))):
        if set(g.symbols.values()) != {"Si1 O4"}:
            continue
        prop_records.append(r)
        silicate_lqgs.append(g)

    # step 4: export graph features of lqg
    lqg_feature_records = export_lqg_feature_record_parallel(lqg_feature_folder, lqgs, prop_records, n_jobs=1)

    # step 5: export features and targets
    if not lqg_feature_records:
        lqg_feature_records = []
        for jf in tqdm(sorted(glob.glob(f"{lqg_feature_folder}/*.json"))):
            zeo_id = os.path.basename(jf).replace(".json", "")
            r = json_load(jf)
            r['zeo_id'] = zeo_id
            lqg_feature_records.append(r)
    df_feat = pd.DataFrame.from_records(lqg_feature_records)
    df_feat.set_index(keys="zeo_id", inplace=True)
    df_feat.sort_index(inplace=True)
    df_target = df_zpp.set_index(keys="zeo_id")
    df_target = df_target.loc[df_feat.index]
    df_feat.to_csv("data_feat.csv")
    df_target.to_csv("data_target.csv")


if __name__ == '__main__':
    main()
