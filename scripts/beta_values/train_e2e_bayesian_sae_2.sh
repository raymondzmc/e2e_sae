#! /bin/bash
export CUDA_VISIBLE_DEVICES=1

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml"
SPARSITY_COEFFS=(0.75 0.8 0.85 0.9 0.95 1.0)
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


# RUN_NAME_PREFIX="e2e_bayesian_beta_0.8"
# for SPARSITY in "${SPARSITY_COEFFS[@]}"
# do
#   python train_sae.py \
#     --config "$CONFIG" \
#     --sparsity_coeff "$SPARSITY" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
#     --wandb_tags "e2e" "bayesian" \
#     --bayesian_sae \
#     --initial_beta 0.8 \
#     --final_beta 0.8
# done

# RUN_NAME_PREFIX="e2e_bayesian_beta_0.7"
# for SPARSITY in "${SPARSITY_COEFFS[@]}"
# do
#   python train_sae.py \
#     --config "$CONFIG" \
#     --sparsity_coeff "$SPARSITY" \
#     --wandb_project "$WANDB_PROJECT" \
#     --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
#     --wandb_tags "e2e" "bayesian" \
#     --bayesian_sae \
#     --initial_beta 0.7 \
#     --final_beta 0.7
# done
