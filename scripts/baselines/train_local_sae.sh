#! /bin/bash
export CUDA_VISIBLE_DEVICES=5

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_local.yaml"
RUN_NAME_PREFIX="local"
SPARSITY_COEFFS=(0.001 0.005 0.008 0.01 0.02 0.05)
LAYERS=(0 1 2 3 5 6 7 8)

for LAYER in "${LAYERS[@]}"
do
    for SPARSITY in "${SPARSITY_COEFFS[@]}"
    do
    python train_sae.py \
        --config "$CONFIG" \
        --sparsity_coeff "$SPARSITY" \
        --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
        --wandb_tags "local" \
        --layer "$LAYER"
    done
done