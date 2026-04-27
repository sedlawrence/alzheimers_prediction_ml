from pathlib import Path
import runpy


_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_TASK1_DL = runpy.run_path(str(_SCRIPTS_DIR / "05_task1_deep_learning.py"))

set_global_seed = _TASK1_DL["set_global_seed"]
resolve_device = _TASK1_DL["resolve_device"]

FeatureStandardizer = _TASK1_DL["FeatureStandardizer"]
RegularizedMLP = _TASK1_DL["RegularizedMLP"]
BottleneckMLP = _TASK1_DL["BottleneckMLP"]
SiameseTimepointMLP = _TASK1_DL["SiameseTimepointMLP"]
GenomeCNN = _TASK1_DL["GenomeCNN"]

make_experiment = _TASK1_DL["make_experiment"]
get_experiments = _TASK1_DL["get_experiments"]
build_model = _TASK1_DL["build_model"]
calculate_metrics = _TASK1_DL["calculate_metrics"]
count_trainable_params = _TASK1_DL["count_trainable_params"]
predict_scores = _TASK1_DL["predict_scores"]
train_one_fold = _TASK1_DL["train_one_fold"]
evaluate_model_cv = _TASK1_DL["evaluate_model_cv"]
summarise_cv_results = _TASK1_DL["summarise_cv_results"]

BATCH_SIZE = _TASK1_DL["BATCH_SIZE"]
MAX_EPOCHS = _TASK1_DL["MAX_EPOCHS"]
PATIENCE = _TASK1_DL["PATIENCE"]
LEARNING_RATE = _TASK1_DL["LEARNING_RATE"]
WEIGHT_DECAY = _TASK1_DL["WEIGHT_DECAY"]
VALIDATION_SIZE = _TASK1_DL["VALIDATION_SIZE"]
N_SPLITS = _TASK1_DL["N_SPLITS"]
N_REPEATS = _TASK1_DL["N_REPEATS"]
RANDOM_STATE = _TASK1_DL["RANDOM_STATE"]
get_effective_n_features = _TASK1_DL["get_effective_n_features"]


def update_runtime_defaults(
    *,
    batch_size=None,
    max_epochs=None,
    patience=None,
    lr=None,
    weight_decay=None,
    val_size=None,
    n_splits=None,
    n_repeats=None,
    random_state=None,
):
    updates = {
        "BATCH_SIZE": batch_size,
        "MAX_EPOCHS": max_epochs,
        "PATIENCE": patience,
        "LEARNING_RATE": lr,
        "WEIGHT_DECAY": weight_decay,
        "VALIDATION_SIZE": val_size,
        "N_SPLITS": n_splits,
        "N_REPEATS": n_repeats,
        "RANDOM_STATE": random_state,
    }

    for key, value in updates.items():
        if value is not None:
            _TASK1_DL[key] = value
            globals()[key] = value
