#!/bin/bash
# Cancel the 3 single-GPU v4 jobs that were submitted before we
# switched to multi-GPU per job.
scancel 485768 485769 485770
sleep 2
squeue -j 485768,485769,485770 2>&1 | tail -5