#! /bin/bash
export CUDA_VISIBLE_DEVICES=6

CONFIG="e2e_sae/scripts/train_tlens_saes/gpt2_e2e_recon_layer4.yaml"
RUN_NAME_PREFIX="ds"
SPARSITY_COEFFS=(30 20 10 5 3 1)

for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
  python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY"
done
