#! /bin/bash
export CUDA_VISIBLE_DEVICES=4

CONFIG="e2e_sae/scripts/train_tlens_saes/gpt2_local_10.yaml"
SPARSITY_COEFFS=(10 5)
# SPARSITY_COEFFS=(10 5 3 1)


for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "ReLU"
done
