#!/bin/bash
# Queue a benchmark on the high-CPU pod from any pod. Returns immediately.
#   submit.sh REF NAME [KWARGS_JSON] [DATASETS]
# Watch: cat ~/work/PairPlex/bench/runs/NAME/STATUS   (queued / running / DONE / DONE_WITH_FAILURES)
# Compare: python diagnosis/bench/compare.py NAME [--baseline v2-base]
set -euo pipefail
REF=$1; NAME=$2; KWARGS=${3:-"{}"}; DATASETS=${4:-sim_cal,sim_imp,real_d02,real_d07}
RUN=/home/jovyan/work/PairPlex/bench/runs/$NAME
if [ -e "$RUN/run.json" ] || grep -qs . "$RUN/STATUS"; then echo "run $NAME exists; pick another NAME" >&2; exit 1; fi
mkdir -p "$RUN"
printf '%s' "$KWARGS" > "$RUN/kwargs.json"
# each job runs its own frozen copy: editing bench.sh later cannot break it (NFS stale handles)
cp /home/jovyan/work/Software/pairplex/diagnosis/bench/bench_job.sh "$RUN/bench.sh"
ssh -n -F /home/jovyan/.ssh/config -o BatchMode=yes highcpu \
  "tmux new-session -d -s bench_$NAME 'bash $RUN/bench.sh $REF $NAME @$RUN/kwargs.json $DATASETS > $RUN/bench.log 2>&1'"
echo "submitted $NAME ($REF); status: $RUN/STATUS"
