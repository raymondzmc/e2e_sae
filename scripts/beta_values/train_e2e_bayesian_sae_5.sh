#! /bin/bash
export CUDA_VISIBLE_DEVICES=4

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml"
SPARSITY_COEFFS=(1.8 1.9 2.0 3.0 4.0 5.0)
WANDB_PROJECT="tinystories-1m-e2e-bayesian-beta-annealing"
RUN_NAME_PREFIX="e2e_bayesian_linear_annealing"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "e2e" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.3 \
    --final_beta 0.3 \
    --initial_sparsity_coeff 0.0 \
    --final_sparsity_coeff "$SPARSITY" \
    --sparsity_coeff_annealing_schedule linear \
    --lr 1e-3
done

# CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml"
# SPARSITY_COEFFS=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2)
# WANDB_PROJECT="tinystories-1m-e2e-bayesian-beta-sweep-no-relu"

# RUN_NAME_PREFIX="e2e_bayesian_beta_0.2"
# for SPARSITY in "${SPARSITY_COEFFS[@]}"
# do
#   python train_sae.py \
#     --config "$CONFIG" \
#     --sparsity_coeff "$SPARSITY" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
#     --wandb_tags "e2e" "bayesian" \
#     --bayesian_sae \
#     --initial_beta 0.2 \
#     --final_beta 0.2
# done

# RUN_NAME_PREFIX="e2e_bayesian_beta_0.1"
# for SPARSITY in "${SPARSITY_COEFFS[@]}"
# do
#   python train_sae.py \
#     --config "$CONFIG" \
#     --sparsity_coeff "$SPARSITY" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
#     --wandb_tags "e2e" "bayesian" \
#     --bayesian_sae \
#     --initial_beta 0.1 \
#     --final_beta 0.1
# done