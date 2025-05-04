#! /bin/bash
export CUDA_VISIBLE_DEVICES=1

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_local_bayesian.yaml"
SPARSITY_COEFFS=(1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5 1e-4)
WANDB_PROJECT="tinystories-1m-local-bayesian-beta-sweep"

RUN_NAME_PREFIX="local_bayesian_beta_0.5"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.5 \
    --final_beta 0.5
done

RUN_NAME_PREFIX="local_bayesian_beta_0.4"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.4 \
    --final_beta 0.4
done

RUN_NAME_PREFIX="local_bayesian_beta_0.3"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.3 \
    --final_beta 0.3
done

RUN_NAME_PREFIX="local_bayesian_beta_0.2"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.2 \
    --final_beta 0.2
done

RUN_NAME_PREFIX="local_bayesian_beta_0.1"
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae \
    --initial_beta 0.1 \
    --final_beta 0.1
done

