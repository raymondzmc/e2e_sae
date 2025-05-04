#! /bin/bash
export CUDA_VISIBLE_DEVICES=2

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_local_bayesian.yaml"
RUN_NAME_PREFIX="local_bayesian"

SPARSITY_COEFFS=(2e-5 3e-5 4e-5 6e-5 7e-5 8e-5 9e-5)
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --wandb_tags "local" "bayesian" \
    --bayesian_sae
done
