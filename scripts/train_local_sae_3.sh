#! /bin/bash
export CUDA_VISIBLE_DEVICES=3

CONFIG="e2e_sae/scripts/train_tlens_saes/gpt2_local.yaml"
SPARSITY_COEFFS=(3 1)


for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "ReLU"
done
