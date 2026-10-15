#!/usr/bin/env bash
set -euo pipefail

# Train the user's VisDrone ControlNet adapter with the Diffusers FLUX example.
# Historical successful configuration: 512px, lr 1e-5, 15,000 steps, validation
# every 500 steps, checkpoints every 1,000 steps, batch 2, accumulation 8,
# cosine schedule, 200 warmup steps, 4 double layers, 0 single layers, seed 42.
#
# Diffusers upstream: https://github.com/huggingface/diffusers
# Revision: UNVERIFIED - requires manual confirmation (local checkout metadata unavailable).
# License evidence: the upstream repository includes an Apache-2.0 LICENSE.
# The Diffusers source tree is a runtime dependency and is not copied here.

usage() {
    cat <<'EOF'
Usage: train_flux_controlnet_visdrone.sh [options]

Required options can also be provided through environment variables:
  --base-model PATH       FLUX.1-dev directory (FLUX_BASE_MODEL)
  --controlnet-model PATH Initial ControlNet directory (CONTROLNET_INIT_MODEL)
  --manifest PATH         VisDrone JSONL manifest (VISDRONE_MANIFEST)
  --output-dir PATH       Checkpoint output directory (OUTPUT_DIR)
  --diffusers-root PATH   Diffusers checkout (DIFFUSERS_ROOT)
  --cache-dir PATH        Runtime cache directory (CACHE_DIR)
  --validation-image PATH Add a validation image; may be supplied twice
  --validation-prompt TEXT Add a prompt matching each validation image
  --resume PATH           Optional explicit checkpoint to resume
  -h, --help              Show this help
EOF
}

BASE_MODEL="${FLUX_BASE_MODEL:-}"
CONTROLNET_MODEL="${CONTROLNET_INIT_MODEL:-}"
MANIFEST="${VISDRONE_MANIFEST:-}"
OUTPUT="${OUTPUT_DIR:-}"
DIFFUSERS="${DIFFUSERS_ROOT:-}"
CACHE="${CACHE_DIR:-}"
RESUME="${RESUME_FROM_CHECKPOINT:-}"
VALIDATION_IMAGES=()
VALIDATION_PROMPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --controlnet-model) CONTROLNET_MODEL="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --output-dir) OUTPUT="$2"; shift 2 ;;
        --diffusers-root) DIFFUSERS="$2"; shift 2 ;;
        --cache-dir) CACHE="$2"; shift 2 ;;
        --validation-image) VALIDATION_IMAGES+=("$2"); shift 2 ;;
        --validation-prompt) VALIDATION_PROMPTS+=("$2"); shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for setting in BASE_MODEL CONTROLNET_MODEL MANIFEST OUTPUT DIFFUSERS CACHE; do
    if [[ -z "${!setting}" ]]; then
        echo "Missing required setting: $setting (use an option or environment variable)" >&2
        exit 2
    fi
done

TRAIN_SCRIPT="$DIFFUSERS/examples/controlnet/train_controlnet_flux.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "Diffusers training script not found: $TRAIN_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 2
fi
mkdir -p "$OUTPUT" "$CACHE"

TRAIN_ARGS=(
    --pretrained_model_name_or_path="$BASE_MODEL"
    --controlnet_model_name_or_path="$CONTROLNET_MODEL"
    --jsonl_for_train="$MANIFEST"
    --conditioning_image_column=conditioning_image
    --image_column=image
    --caption_column=text
    --output_dir="$OUTPUT"
    --mixed_precision=bf16
    --resolution=512
    --learning_rate=1e-5
    --max_train_steps=15000
    --validation_steps=500
    --checkpointing_steps=1000
    --train_batch_size=2
    --gradient_accumulation_steps=8
    --report_to=tensorboard
    --lr_scheduler=cosine
    --lr_warmup_steps=200
    --num_double_layers=4
    --num_single_layers=0
    --seed=42
    --dataloader_num_workers=4
    --allow_tf32
    --gradient_checkpointing
    --checkpoints_total_limit=5
    --cache_dir="$CACHE"
    --tracker_project_name=flux_controlnet_visdrone
    --proportion_empty_prompts=0.1
)

if [[ ${#VALIDATION_IMAGES[@]} -gt 0 ]]; then
    TRAIN_ARGS+=(--validation_image "${VALIDATION_IMAGES[@]}")
fi
if [[ ${#VALIDATION_PROMPTS[@]} -gt 0 ]]; then
    TRAIN_ARGS+=(--validation_prompt "${VALIDATION_PROMPTS[@]}")
fi
if [[ -n "$RESUME" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint="$RESUME")
fi

echo "Launching Diffusers FLUX ControlNet training. Existing checkpoints are preserved."
exec accelerate launch "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
