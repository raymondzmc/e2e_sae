#! /bin/bash
export CUDA_VISIBLE_DEVICES=2

CONFIG="e2e_sae/scripts/train_tlens_saes/gpt2_e2e_recon.yaml"
RUN_NAME_PREFIX="ds"
SPARSITY_COEFFS=(50 30 20 10 5 3 1)

for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY"
done
