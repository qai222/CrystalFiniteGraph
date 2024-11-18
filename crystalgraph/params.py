from pymatgen.analysis.local_env import CrystalNN

_default_CNN = CrystalNN(
    weighted_cn=False,
    cation_anion=False,
    distance_cutoffs=(0.5, 1),
    x_diff_weight=3.0,
    porous_adjustment=True,
)

ENV_CRYSTALGRAPH_DATA_DIR = "CRYSTALGRAPH_DATA_DIR"
