#!/bin/bash
# Submit the FULL (nonlinear) DGP experiment matrix.
#
# Mirrors the prior simple-DGP submission (results/{scale}_simple_*) exactly,
# but drops --simple_dgp so the nonlinear DGP is used:
#     y = b0 + (b1*X1 + X1^2) + (b2*X2 + 2*X2) + eps
# aggregate_metrics.py maps the non-simple DGP to the name token "_full_",
# so output dirs are named results/{scale}_full_* to aggregate cleanly.
#
# Conditions per scale (same 5 the simple run used):
#   baseline             : default dim, coords, untrained   (dir-labeled dim0/baseline)
#   dim8                 : --embed_dim 8                     (untrained, emb+coords)
#   embonly_dim8         : --embed_dim 8 --no_coords         (untrained, emb only)
#   trained_dim8         : --embed_dim 8 --train_encoder     (contrastive, emb+coords)
#   trained_embonly_dim8 : --embed_dim 8 --no_coords --train_encoder
#
# Each line submits a 12-task array (encoders 0-11) x 25 reps via run_experiments.bash.

set -e
cd /u/dkiv2/group_dkiv2/active/loc-enc-svcs

declare -a CONDS=(
  "baseline|"
  "dim8|--embed_dim 8"
  "embonly_dim8|--embed_dim 8 --no_coords"
  "trained_dim8|--embed_dim 8 --train_encoder"
  "trained_embonly_dim8|--embed_dim 8 --no_coords --train_encoder"
)

for SCALE in grid county global; do
  # global explains 10k points through GeoShapley (~2.2h/encoder); give it headroom.
  if [ "$SCALE" = "global" ]; then
    RES="--mem=32G --time=10:00:00"
  else
    RES=""
  fi
  for C in "${CONDS[@]}"; do
    SUFFIX="${C%%|*}"
    FLAGS="${C#*|}"
    OUT="results/${SCALE}_full_${SUFFIX}"
    echo "SUBMIT ${SCALE}_full_${SUFFIX}  flags=[${FLAGS}]  -> ${OUT}"
    sbatch $RES --job-name="${SCALE}_full_${SUFFIX}" \
      run_experiments.bash "$SCALE" "$OUT" MLP 25 "$FLAGS"
  done
done

echo "All full-DGP jobs submitted."
