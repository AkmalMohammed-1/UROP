#!/bin/bash
# set_ppdd_hparams.sh
# Usage:
#   ./set_ppdd_hparams.sh <lambda_cos> <lambda_calib> <lambda_div> [debug]
#
# Example (your first sweep step):
#   ./set_ppdd_hparams.sh 1.0 1.0 0.0 true
#
# Edits config/ipc10/cifar10.yaml in place. If a key already exists it is
# updated; if not, it's appended with 2-space indent to match the rest of
# the "condense:" block. Prints the final values so you can eyeball them
# before launching.

set -e

CONFIG="/media/cs18s504/DATA/Srikiran/MMD_PPDD_Aggressive/config/ipc10/cifar10.yaml"

LAMBDA_COS="$1"
LAMBDA_CALIB="$2"
LAMBDA_DIV="$3"
DEBUG="${4:-true}"

if [ -z "$LAMBDA_COS" ] || [ -z "$LAMBDA_CALIB" ] || [ -z "$LAMBDA_DIV" ]; then
    echo "Usage: $0 <lambda_cos> <lambda_calib> <lambda_div> [debug=true|false]"
    exit 1
fi

set_or_append() {
    local key="$1"
    local value="$2"
    if grep -q "^  ${key}:" "$CONFIG"; then
        # key exists -- update in place, preserve 2-space indent
        sed -i "s/^  ${key}:.*/  ${key}: ${value}/" "$CONFIG"
    else
        # key missing -- append under the condense: block (2-space indent)
        echo "  ${key}: ${value}" >> "$CONFIG"
    fi
}

set_or_append "ppdd_lambda_cos"   "$LAMBDA_COS"
set_or_append "ppdd_lambda_calib" "$LAMBDA_CALIB"
set_or_append "ppdd_lambda_div"   "$LAMBDA_DIV"
set_or_append "ppdd_debug"        "$DEBUG"

echo "Updated $CONFIG:"
grep -n "ppdd_lambda_cos\|ppdd_lambda_calib\|ppdd_lambda_div\|ppdd_debug" "$CONFIG"