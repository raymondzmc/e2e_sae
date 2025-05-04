#! /bin/bash
export CUDA_VISIBLE_DEVICES=2

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml"
SPARSITY_COEFFS=(1.05 1.1 1.15 1.2 1.25 1.3)
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

# export CUDA_VISIBLE_DEVICES=4

# CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml"
# SPARSITY_COEFFS=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2)
# WANDB_PROJECT="tinystories-1m-e2e-bayesian-beta-sweep-no-relu"
# RUN_NAME_PREFIX="e2e_bayesian_beta_0.6"
# for SPARSITY in "${SPARSITY_COEFFS[@]}"
# do
#   python train_sae.py \
#     --config "$CONFIG" \
#     --sparsity_coeff "$SPARSITY" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
#     --wandb_tags "e2e" "bayesian" \
#     --bayesian_sae \
#     --initial_beta 0.6 \
#     --final_beta 0.6
# done

# RUN_NAME_PREFIX="e2e_bayesian_beta_0.5"
# for SPARSITY in "${SPARSITY_COEFFS[@]}"
# do
#   python train_sae.py \
#     --config "$CONFIG" \
#     --sparsity_coeff "$SPARSITY" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
#     --wandb_tags "e2e" "bayesian" \
#     --bayesian_sae \
#     --initial_beta 0.5 \
#     --final_beta 0.5
# done
