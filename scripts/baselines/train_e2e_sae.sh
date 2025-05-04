#! /bin/bash
export CUDA_VISIBLE_DEVICES=4

CONFIG="e2e_sae/scripts/train_tlens_saes/tinystories_1M_e2e.yaml"
RUN_NAME_PREFIX="e2e"
SPARSITY_COEFFS=(50 30 20 10 5 3 1)
LAYERS=(0 1 2 3 5 6 7 8)

for LAYER in "${LAYERS[@]}"
do
  for SPARSITY in "${SPARSITY_COEFFS[@]}"
  do
    python train_sae.py \
      --config "$CONFIG" \
      --sparsity_coeff "$SPARSITY" \
      --wandb_run_name_prefix "$RUN_NAME_PREFIX" \
      --wandb_tags "e2e" \
      --layer "$LAYER"
  done
done
