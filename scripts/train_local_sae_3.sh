#! /bin/bash
export CUDA_VISIBLE_DEVICES=7

CONFIG="e2e_sae/scripts/train_tlens_saes/gpt2_local_10.yaml"
SPARSITY_COEFFS=(0.01 0.005)


for SPARSITY in "${SPARSITY_COEFFS[@]}"
do
python train_sae.py \
    --config "$CONFIG" \
    --sparsity_coeff "$SPARSITY" \
    --wandb_run_name_prefix "ReLU"
done
