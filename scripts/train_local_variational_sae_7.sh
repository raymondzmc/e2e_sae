#! /bin/bash
export CUDA_VISIBLE_DEVICES=7

CONFIG="e2e_sae/scripts/train_tlens_saes/gpt2_local_bayesian.yaml"
RUN_NAME_PREFIX="Variational"

SPARSITY_COEFFS=(4 3)
for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
    --bayesian_sae
done
