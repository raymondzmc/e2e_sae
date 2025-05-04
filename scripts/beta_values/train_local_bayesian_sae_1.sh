#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_local_bayesian.yaml"
SPARSITY_COEFFS=(1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5 1e-4)
WANDB_PROJECT="tinystories-1m-local-bayesian-beta-sweep"
RUN_NAME_PREFIX="local_bayesian_beta_1"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 1.0 \
    --final_beta 1.0
done

RUN_NAME_PREFIX="local_bayesian_beta_0.9"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.9 \
    --final_beta 0.9
done

RUN_NAME_PREFIX="local_bayesian_beta_0.8"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.8 \
    --final_beta 0.8
done

RUN_NAME_PREFIX="local_bayesian_beta_0.7"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.7 \
    --final_beta 0.7
done

RUN_NAME_PREFIX="local_bayesian_beta_0.6"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.6 \
    --final_beta 0.6
done

