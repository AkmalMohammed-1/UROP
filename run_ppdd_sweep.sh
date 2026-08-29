#!/bin/bash
# run_ppdd_sweep.sh
#
# For each (lambda_cos, lambda_calib, lambda_div) combo:
#   1. Set hyperparameters in cifar10.yaml (debug forced OFF, see note below)
#   2. Run condensation from CONDENSE_DIR
#   3. Identify the run directory this specific condensation run created,
#      by diffing a before/after snapshot of the results dir (never just
#      "newest by mtime" -- see find_new_run_dir())
#   4. Verify distilled_data/data_20000.pt exists in that directory
#   5. Run evaluation on that exact checkpoint
#   6. Extract Mean Accuracy / Std Deviation / All result from the
#      EVALUATION log only
#   7. Append a one-line summary to results_sweep.txt
#   8. On any failure at steps 2-6, record it clearly and move on to the
#      next combo -- never silently skip a step or evaluate the wrong
#      checkpoint.
#
# Usage:
#   chmod +x run_ppdd_sweep.sh
#   tmux new -s ppdd_sweep
#   ./run_ppdd_sweep.sh
#   (Ctrl+B then D to detach; `tmux attach -t ppdd_sweep` to check back in)

# Intentionally NOT using `set -e` globally: condensation/evaluation runs
# are expected to sometimes fail, and we need to catch that ourselves,
# log it, and move on to the next combo rather than have the whole sweep
# die on the first bad run.

NCFM_ROOT="/media/cs18s504/DATA/Srikiran/MMD_PPDD_Aggressive"
CONDENSE_DIR="${NCFM_ROOT}/condense"
EVAL_DIR="${NCFM_ROOT}/evaluation"
HPARAM_SCRIPT="${NCFM_ROOT}/set_ppdd_hparams.sh"
RESULTS_FILE="${NCFM_ROOT}/results_sweep.txt"
CONDENSE_LOG_DIR="${NCFM_ROOT}/sweep_logs/condense"
EVAL_LOG_DIR="${NCFM_ROOT}/sweep_logs/evaluate"
RUNS_BASE_DIR="${NCFM_ROOT}/results/condense/condense/cifar10/ipc10"

mkdir -p "$CONDENSE_LOG_DIR" "$EVAL_LOG_DIR"

# --- Combos to run: (lambda_cos  lambda_calib  lambda_div) ---
# lambda_cos held at 0.0 (removed). Sweeping higher lambda_calib and lambda_div.
COMBOS=(
    "0.0 2.0 1.0"
    "0.0 3.0 1.0"
    "0.0 4.0 1.0"
    "0.0 2.0 1.5"
    "0.0 3.0 1.5"
    "0.0 4.0 1.5"
)

CONDENSE_CMD='CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --nnodes=1 --master_port=34153 condense_script.py --gpu="0" --ipc=10 --config_path=../config/ipc10/cifar10.yaml'

# --load_path is filled in per-combo once the checkpoint is known
EVAL_CMD_TEMPLATE='CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --nnodes=1 --master_port=34154 evaluation_script.py --gpu="0" --ipc=10 --config_path=../config/ipc10/cifar10.yaml --load_path="__LOAD_PATH__"'

log_result() {
    # Always goes to both the terminal/tmux pane and the results file.
    echo "$1" | tee -a "$RESULTS_FILE"
}

# Snapshot the run-directory listing under RUNS_BASE_DIR. Used both
# before and after condensation so we can diff them -- this is what lets
# us find the correct new run dir even if RUNS_BASE_DIR already has many
# old runs in it (robust to a pre-existing results directory).
snapshot_run_dirs() {
    if [ -d "$RUNS_BASE_DIR" ]; then
        find "$RUNS_BASE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
    fi
    # if RUNS_BASE_DIR doesn't exist yet, this prints nothing -- fine,
    # "before" snapshot is just empty in that case.
}

# Diff before/after snapshots to find the run dir(s) created by this
# specific condensation call. Never falls back to "just pick newest" --
# if the diff is ambiguous, that's treated as a failure so we don't risk
# evaluating the wrong checkpoint.
find_new_run_dir() {
    local before_file="$1"
    local after_file="$2"

    comm -13 "$before_file" "$after_file"
}

run_one_combo() {
    local LCOS="$1"
    local LCALIB="$2"
    local LDIV="$3"

    local TAG="cos${LCOS}_calib${LCALIB}_div${LDIV}"
    local CONDENSE_LOG="${CONDENSE_LOG_DIR}/${TAG}.log"
    local EVAL_LOG="${EVAL_LOG_DIR}/${TAG}.log"

    echo ""
    echo "==================== Combo: lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV ===================="

    # --- Step 1: set hyperparameters, debug ALWAYS forced false ------------
    # Note: set_ppdd_hparams.sh's own default for the debug arg is "true".
    # We override that here unconditionally -- the sweep must never enable
    # debug logging, regardless of what the hparam script's default is.
    echo ">>> Setting hyperparameters (debug forced OFF)"
    if ! "$HPARAM_SCRIPT" "$LCOS" "$LCALIB" "$LDIV" false; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (could not set hyperparameters, see terminal output above)"
        return
    fi

    # --- Step 2: snapshot run dirs BEFORE condensation ----------------------
    local before_snapshot
    before_snapshot=$(mktemp)
    snapshot_run_dirs > "$before_snapshot"

    # --- Step 3: run condensation --------------------------------------------
    echo ">>> Running condensation, logging to $CONDENSE_LOG"
    cd "$CONDENSE_DIR" || {
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (could not cd into $CONDENSE_DIR)"
        rm -f "$before_snapshot"
        return
    }

    eval "$CONDENSE_CMD" > "$CONDENSE_LOG" 2>&1
    local condense_status=$?

    if [ $condense_status -ne 0 ]; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (condensation exited with status $condense_status, see $CONDENSE_LOG)"
        rm -f "$before_snapshot"
        return
    fi

    # --- Step 4: identify the new run directory ------------------------------
    local after_snapshot
    after_snapshot=$(mktemp)
    snapshot_run_dirs > "$after_snapshot"

    local new_dirs
    new_dirs=$(find_new_run_dir "$before_snapshot" "$after_snapshot")
    rm -f "$before_snapshot" "$after_snapshot"

    local new_dir_count
    new_dir_count=$(echo "$new_dirs" | grep -c . || true)

    if [ "$new_dir_count" -eq 0 ]; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (condensation reported success but no new run directory appeared under $RUNS_BASE_DIR; see $CONDENSE_LOG)"
        return
    elif [ "$new_dir_count" -gt 1 ]; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (ambiguous: $new_dir_count new run directories appeared -- refusing to guess which one belongs to this run: $new_dirs)"
        return
    fi

    local RUN_DIR="${RUNS_BASE_DIR}/${new_dirs}"
    echo ">>> New run directory identified: $RUN_DIR"

    # --- Step 5: verify checkpoint exists -------------------------------------
    local CHECKPOINT="${RUN_DIR}/distilled_data/data_20000.pt"
    if [ ! -f "$CHECKPOINT" ]; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (data_20000.pt missing at $CHECKPOINT, refusing to evaluate)"
        return
    fi
    echo ">>> Checkpoint confirmed: $CHECKPOINT"

    # --- Step 6: run evaluation on that exact checkpoint ----------------------
    echo ">>> Running evaluation, logging to $EVAL_LOG"
    cd "$EVAL_DIR" || {
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (could not cd into $EVAL_DIR)"
        return
    }

    local EVAL_CMD="${EVAL_CMD_TEMPLATE//__LOAD_PATH__/$CHECKPOINT}"
    eval "$EVAL_CMD" > "$EVAL_LOG" 2>&1
    local eval_status=$?

    if [ $eval_status -ne 0 ]; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (evaluation exited with status $eval_status, see $EVAL_LOG; checkpoint was $CHECKPOINT)"
        return
    fi

    # --- Step 7: extract results from the EVALUATION log (not condense log) --
    local ACC_LINE
    ACC_LINE=$(grep "Mean Accuracy" "$EVAL_LOG" | tail -n 1)

    if [ -z "$ACC_LINE" ]; then
        log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> FAILED (evaluation completed but no 'Mean Accuracy' line found in $EVAL_LOG; checkpoint was $CHECKPOINT)"
        return
    fi

    log_result "lambda_cos=$LCOS lambda_calib=$LCALIB lambda_div=$LDIV -> $ACC_LINE  [checkpoint: $CHECKPOINT]"
}

echo "==================== PPDD sweep started $(date) ====================" >> "$RESULTS_FILE"

for combo in "${COMBOS[@]}"; do
    read -r LCOS LCALIB LDIV <<< "$combo"
    run_one_combo "$LCOS" "$LCALIB" "$LDIV"
done

echo "==================== PPDD sweep finished $(date) ====================" >> "$RESULTS_FILE"
echo ""
echo "All combos processed. Summary:"
cat "$RESULTS_FILE"