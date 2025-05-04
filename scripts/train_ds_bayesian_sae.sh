#! /bin/bash
export CUDA_VISIBLE_DEVICES=3

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e_recon_bayesian.yaml"
RUN_NAME_PREFIX="ds_bayesian"

SPARSITY_COEFFS=(0.001 0.005 0.008 0.01 0.02 0.05)

for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --bayesian_sae \
    --wandb_tags "ds" "bayesian"
done
