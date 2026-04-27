#!/usr/bin/env bash

set -euo pipefail

cd scripts

###############################################################################
# Task 1
###############################################################################

# 1) Task 1 EDA
python 01_describe_array.py

# 2) Task 1 baselines
python 02_task1_baselines.py

# 3) Task 1 XGBoost
python 03_task1_xgboost.py

# 4) Task 1 SVM + linear models
python 04_task1_svm_and_linear_models.py

# 5) Task 1 deep learning experiments
# Full preset:
python 05_task1_deep_learning.py --device mps

# Run individual experiments into separate output folders:
python 05_task1_deep_learning.py --device mps --experiments mlp_t1_only --run_subdir dl_mlp_t1
python 05_task1_deep_learning.py --device cpu --experiments siamese_mlp_t0_t1 --run_subdir dl_siamese_cpu
python 05_task1_deep_learning.py --device mps --experiments bottleneck_mlp_t0_t1_delta --run_subdir dl_bottleneck_t0_t1_delta
python 05_task1_deep_learning.py --device mps --experiments genome_cnn_t0_t1 --run_subdir dl_cnn_pair
python 05_task1_deep_learning.py --device mps --experiments genome_cnn_t0_t1_delta --run_subdir dl_cnn_delta

# Tuned small last-pass preset:
python 05_task1_deep_learning.py --preset tuned_small --device mps --run_subdir dl_tuned_small

# Tuned small experiments run separately:
python 05_task1_deep_learning.py --preset tuned_small --device mps --experiments mlp_t1_only_wide --run_subdir dl_tuned_mlp_wide
python 05_task1_deep_learning.py --preset tuned_small --device mps --experiments mlp_t1_only_narrow --run_subdir dl_tuned_mlp_narrow
python 05_task1_deep_learning.py --preset tuned_small --device mps --experiments bottleneck_mlp_t1_only_wide --run_subdir dl_tuned_bottleneck_wide
python 05_task1_deep_learning.py --preset tuned_small --device mps --experiments bottleneck_mlp_t1_only_narrow --run_subdir dl_tuned_bottleneck_narrow
python 05_task1_deep_learning.py --preset tuned_small --device mps --experiments siamese_mlp_t0_t1_small --run_subdir dl_tuned_siamese_small
python 05_task1_deep_learning.py --preset tuned_small --device mps --experiments siamese_mlp_t0_t1_wide --run_subdir dl_tuned_siamese_wide

###############################################################################
# Task 2
###############################################################################

# 6) Task 2 EDA
python 06_task2_describe_array.py

# 7) Task 2 baselines
python 07_task2_baselines.py

# 8) Task 2 XGBoost
python 08_task2_xgboost.py

# 9) Task 2 SVM + linear models
python 09_task2_svm_and_linear_models.py

# 10) Task 2 deep learning experiments
# Full preset:
python 10_task2_deep_learning.py --device mps

# Run individual experiments into separate output folders:
python 10_task2_deep_learning.py --device mps --experiments mlp_t1_only --run_subdir dl_mlp_t1
python 10_task2_deep_learning.py --device cpu --experiments siamese_mlp_t0_t1 --run_subdir dl_siamese_cpu
python 10_task2_deep_learning.py --device mps --experiments bottleneck_mlp_t0_t1_delta --run_subdir dl_bottleneck_t0_t1_delta
python 10_task2_deep_learning.py --device mps --experiments genome_cnn_t0_t1 --run_subdir dl_cnn_pair
python 10_task2_deep_learning.py --device mps --experiments genome_cnn_t0_t1_delta --run_subdir dl_cnn_delta

# Tuned small last-pass preset:
python 10_task2_deep_learning.py --preset tuned_small --device mps --run_subdir dl_tuned_small

# Tuned small experiments run separately:
python 10_task2_deep_learning.py --preset tuned_small --device mps --experiments mlp_t1_only_wide --run_subdir dl_tuned_mlp_wide
python 10_task2_deep_learning.py --preset tuned_small --device mps --experiments mlp_t1_only_narrow --run_subdir dl_tuned_mlp_narrow
python 10_task2_deep_learning.py --preset tuned_small --device mps --experiments bottleneck_mlp_t1_only_wide --run_subdir dl_tuned_bottleneck_wide
python 10_task2_deep_learning.py --preset tuned_small --device mps --experiments bottleneck_mlp_t1_only_narrow --run_subdir dl_tuned_bottleneck_narrow
python 10_task2_deep_learning.py --preset tuned_small --device mps --experiments siamese_mlp_t0_t1_small --run_subdir dl_tuned_siamese_small
python 10_task2_deep_learning.py --preset tuned_small --device mps --experiments siamese_mlp_t0_t1_wide --run_subdir dl_tuned_siamese_wide

###############################################################################
# Joint DL optimisation for Tasks 1 and 2
###############################################################################

python 11_task1_task2_dl_optimisation.py --device mps

python 12_make_report_figures.py