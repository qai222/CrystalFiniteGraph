import os.path
import time

from loguru import logger

from crystalgraph.benchmark import LqgXGBoost, BenchmarkDataset, LqgXGBoostParams, DummyModel, DummyModelParams, StructCgcnn, StructCgcnnParams, StructCgcnnResults

ZeoliteTargets = (
    "largest_included_sphere, "
    "largest_free_sphere, "
    "largest_included_sphere_along_free_sphere_path, "
    "Unitcell_volume, "
    "Density, ASA_A^2"
)

ZeoliteTargets = [t.strip() for t in ZeoliteTargets.split(",")]


def train_xgboost(dataset: BenchmarkDataset):
    model_params = LqgXGBoostParams()
    model = LqgXGBoost(
        dataset=dataset,
        work_dir=f"{dataset.dataset_name}/{dataset.target_name}/{LqgXGBoost.__name__}",
        benchmark_model_params=model_params,
    )
    logger.add(os.path.join(model.work_dir, "train.log"))
    model.train_and_eval(verbose=0)


def train_dummy(dataset: BenchmarkDataset):
    model_params = DummyModelParams()
    model = DummyModel(
        dataset=dataset,
        work_dir=f"{dataset.dataset_name}/{dataset.target_name}/{DummyModel.__name__}",
        benchmark_model_params=model_params,
    )
    logger.add(os.path.join(model.work_dir, "train.log"))
    model.train_and_eval(verbose=0)

def train_cgcnn(dataset: BenchmarkDataset):
    model_params = StructCgcnnParams(
        batch_size=128
    )
    model = StructCgcnn(
        dataset=dataset,
        work_dir=f"{dataset.dataset_name}/{dataset.target_name}/{StructCgcnn.__name__}",
        benchmark_model_params=model_params,
        structure_file_extension="cssr",
        structure_file_folder_path="../data/zeolite/cssr"
    )
    logger.add(os.path.join(model.work_dir, "train.log"))
    model.train_and_eval()


def train_all(training_functions: list):
    for train_function in training_functions:
        for target_name in ZeoliteTargets:
            logger.info(f"working on: '{target_name}'")
            ts1 = time.perf_counter()
            dataset = BenchmarkDataset(
                dataset_name="zeolite",
                dataset_path="../data/zeolite",
                structure_extension="cssr",
                structure_folder_name="cssr",
                lqg_feat_csv="data_feat.csv",
                target_csv="data_target.csv",
                target_name=target_name
            )
            train_function(dataset)
            ts2 = time.perf_counter()
            logger.info(f"took: {ts2 - ts1} s")


if __name__ == '__main__':
    train_all(
        [
            # train_dummy,
            # train_xgboost,
            train_cgcnn,
        ]
    )
