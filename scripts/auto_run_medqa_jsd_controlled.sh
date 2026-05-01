#!/bin/bash
# ABOUTME: Automated MedQA JSD Controlled Perturbation production run with automatic relaunch
# ABOUTME: Monitors jobs and automatically relaunches until all GPUs complete

CHECKPOINT_DIR="checkpoints/medqa_jsd_controlled"
MODEL="deepseek_r1_0528"
TOTAL_GPUS=4
SBATCH_SCRIPT="slurm_jobs/run_medqa_jsd_controlled.sbatch"
JOB_NAME="medqa_jsd_ctrl"
CHECK_INTERVAL=300  # Check every 5 minutes

echo "=========================================="
echo "MedQA JSD Controlled Perturbation Run"
echo "=========================================="
echo "Model: ${MODEL}"
echo "GPUs: ${TOTAL_GPUS}"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Function to check if analysis is complete
check_completion() {
    # Check if final output file exists (merge already done)
    if [ -f "results/medqa_jsd_controlled_analysis_${MODEL}.xlsx" ]; then
        return 0  # Complete
    fi

    # Check if all GPU completion markers exist
    COMPLETE_COUNT=$(ls -1 ${CHECKPOINT_DIR}/${MODEL}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
    if [ "$COMPLETE_COUNT" -eq $TOTAL_GPUS ]; then
        return 0  # All complete (merge may be in progress)
    fi

    return 1  # Not complete
}

# Function to check if jobs are running
check_jobs_running() {
    RUNNING_JOBS=$(squeue -u $USER -n ${JOB_NAME} -h | wc -l)
    if [ "$RUNNING_JOBS" -gt 0 ]; then
        return 0  # Jobs running
    else
        return 1  # No jobs running
    fi
}

# Main loop
ITERATION=1
while true; do
    echo "[$(date)] Iteration ${ITERATION}"

    # Check if all GPUs complete
    if check_completion; then
        COMPLETE_COUNT=$(ls -1 ${CHECKPOINT_DIR}/${MODEL}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
        echo "All ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete!"
        echo "Results: results/medqa_jsd_controlled_analysis_${MODEL}.xlsx"
        echo ""
        echo "=========================================="
        echo "Production run finished successfully!"
        echo "Completed at: $(date)"
        echo "=========================================="
        exit 0
    fi

    # Check progress
    COMPLETE_COUNT=$(ls -1 ${CHECKPOINT_DIR}/${MODEL}_gpu*_of_${TOTAL_GPUS}_COMPLETE 2>/dev/null | wc -l)
    echo "Progress: ${COMPLETE_COUNT}/${TOTAL_GPUS} GPUs complete"

    # Check if jobs are running
    if check_jobs_running; then
        echo "Jobs currently running:"
        squeue -u $USER -n ${JOB_NAME}
        echo "Waiting ${CHECK_INTERVAL}s before next check..."
        sleep $CHECK_INTERVAL
    else
        echo "No jobs running - launching/relaunching..."

        # Submit job
        SUBMITTED_JOB_ID=$(sbatch ${SBATCH_SCRIPT} | awk '{print $NF}')
        echo "Submitted job: ${SUBMITTED_JOB_ID}"

        # Wait a bit for jobs to start
        sleep 60

        # Check if jobs started successfully
        if check_jobs_running; then
            echo "Jobs started successfully"
            squeue -u $USER -n ${JOB_NAME}
        else
            echo "Warning: Jobs may not have started"
            squeue -u $USER -n ${JOB_NAME}
        fi

        echo "Waiting ${CHECK_INTERVAL}s before next check..."
        sleep $CHECK_INTERVAL
    fi

    ITERATION=$((ITERATION + 1))
done
