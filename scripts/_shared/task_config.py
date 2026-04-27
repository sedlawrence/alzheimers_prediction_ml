from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 3
DEFAULT_TOP_N_CPGS = 200

ANNOTATION_PATH = Path("../data/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    h5_path: Path
    out_dir: Path
    class0_key: str
    class1_key: str
    cpg_key: str
    class0_label: str
    class1_label: str


TASK1_SPEC = TaskSpec(
    name="task1",
    h5_path=Path("../data/temporal_two_sets_n2000.h5"),
    out_dir=Path("../results/task1"),
    class0_key="X_cn_to_cn",
    class1_key="X_cn_to_mci",
    cpg_key="cpg_ids_cn",
    class0_label="Control -> Control",
    class1_label="Control -> MCI",
)

TASK2_SPEC = TaskSpec(
    name="task2",
    h5_path=Path("../data/temporal_two_sets_n2000.h5"),
    out_dir=Path("../results/task2"),
    class0_key="X_mci_to_mci",
    class1_key="X_mci_to_dem",
    cpg_key="cpg_ids_mci",
    class0_label="MCI -> MCI",
    class1_label="MCI -> AD",
)
