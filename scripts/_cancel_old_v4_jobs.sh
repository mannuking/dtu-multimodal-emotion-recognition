#!/bin/bash
# Cancel ALL currently-running dtu-ser-v4 jobs for our user.
# Works regardless of how many were submitted (1, 3, or 30).
# Prints which ones got cancelled so we know what to re-submit.
#
# Run from HPC login node (safe — pure scancel, no compute).

echo "Cancelling all running dtu-ser-v4 jobs for $USER..."
RUNNING=$(squeue -u $USER -t R -o "%.10i" -h | tr '\n' ' ')
PENDING=$(squeue -u $USER -t PD -o "%.10i" -h | tr '\n' ' ')
ALL="$RUNNING $PENDING"
ALL=$(echo $ALL | xargs)  # trim whitespace

if [ -z "$ALL" ]; then
    echo "No jobs to cancel. squeue shows nothing for $USER."
    exit 0
fi

echo "About to cancel: $ALL"
echo $ALL | xargs scancel
sleep 2

echo ""
echo "After cancel:"
squeue -u $USER
echo ""
echo "Done. Re-submit with:"
echo "  sbatch scripts/train_ser_v4.sbatch 42"
echo "  sbatch scripts/train_ser_v4.sbatch 43"
echo "  sbatch scripts/train_ser_v4.sbatch 44"