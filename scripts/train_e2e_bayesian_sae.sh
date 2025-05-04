#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_bayesian.yaml"
RUN_NAME_PREFIX="e2e_bayesian"

SPARSITY_COEFFS=(0.001 0.005 0.008 0.01 0.02 0.05 0.08 0.1 0.2 0.5)

for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "e2e" "bayesian" \
    --bayesian_sae
done
