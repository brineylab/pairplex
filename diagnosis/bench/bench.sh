#!/bin/bash
# PairPlex v2 benchmark: run one git ref on the frozen datasets and score it.
# Runs ON the high-CPU pod (use submit.sh from anywhere). At most 2 benchmarks run at once
# (slot locks); each runs its 4 datasets in parallel with 30 workers each (high-CPU pod is ours: 240 cores, 1 TB).
#
#   bench.sh REF NAME [KWARGS_JSON] [DATASETS]
#     REF       branch, tag or commit in ~/work/Software/pairplex (local refs, then origin/REF)
#     NAME      run name -> ~/work/PairPlex/bench/runs/NAME/
#     KWARGS    JSON passed to pairplex.pairplex.run (default {}), or @path to a JSON file
#     DATASETS  comma list from: sim_cal,sim_imp,real_d02,real_d07 (default: all)
set -uo pipefail
REF=$1; NAME=$2; KWARGS=${3:-"{}"}; DATASETS=${4:-sim_cal,sim_imp,real_d02,real_d07}
[[ $KWARGS == @* ]] && KWARGS=$(cat "${KWARGS#@}")   # @file form (used by submit.sh)
REPO=/home/jovyan/work/Software/pairplex
HARNESS=$REPO/diagnosis/bench          # scoring code: the diagnosis checkout
PY=/home/jovyan/work/pairplex-simplex/.venv/bin/python
BENCH=/home/jovyan/work/PairPlex/bench
SIM=/home/jovyan/work/PairPlex/diagnosis/results/simplex_20k96
REAL=/home/jovyan/fast/imm/pairplex_bench/real
RUN=$BENCH/runs/$NAME
mkdir -p $RUN
echo "queued $(date -u +%FT%TZ)" > $RUN/STATUS

# ---- wait for a slot ----------------------------------------------------------------------
exec 8>/tmp/pairplex_bench_slot1.lock 9>/tmp/pairplex_bench_slot2.lock
while true; do
  flock -n 8 && { SLOT=1; break; }
  flock -n 9 && { SLOT=2; break; }
  sleep 30
done
echo "running slot$SLOT $(date -u +%FT%TZ)" > $RUN/STATUS

# ---- snapshot the code under test ---------------------------------------------------------
SHA=$(git -C $REPO rev-parse --verify -q "$REF^{commit}" || git -C $REPO rev-parse --verify -q "origin/$REF^{commit}")
if [ -z "$SHA" ]; then echo "FAILED: unknown ref $REF" > $RUN/STATUS; exit 1; fi
SRC=$BENCH/src/$SHA
if [ ! -d $SRC ]; then
  rm -rf $SRC.tmp; mkdir -p $SRC.tmp && git -C $REPO archive $SHA | tar --no-same-owner -x -C $SRC.tmp && mv $SRC.tmp $SRC
fi
if [ ! -d $SRC ]; then echo "FAILED: could not snapshot $SHA" > $RUN/STATUS; exit 1; fi
cat > $RUN/run.json <<EOF
{"ref": "$REF", "sha": "$SHA", "kwargs": $KWARGS, "datasets": "$DATASETS", "slot": $SLOT,
 "started_utc": "$(date -u +%FT%TZ)", "harness_sha": "$(git -C $REPO rev-parse HEAD)"}
EOF

# ---- datasets -----------------------------------------------------------------------------
export PAIRPLEX_MAX_WORKERS=30 OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_one() {
  local ds=$1 reads kind truth=""
  case $ds in
    sim_cal)  reads=$SIM/calibrated/sim/reads;        truth=$SIM/calibrated/sim/truth;        kind=sim ;;
    sim_imp)  reads=$SIM/improved_protocol/sim/reads; truth=$SIM/improved_protocol/sim/truth; kind=sim ;;
    real_d02) reads=$REAL/donor02_memory; kind=real ;;
    real_d07) reads=$REAL/donor07_memory; kind=real ;;
    *) echo "unknown dataset $ds"; return 1 ;;
  esac
  local out=$RUN/$ds
  rm -rf $out; mkdir -p $out
  local t0=$(date +%s)
  ( cd $SRC && $PY -W ignore $HARNESS/stage_run.py $SRC $reads $out/pp --kwargs "$KWARGS" ) > $out/run.log 2>&1 || { echo FAILED > $out/STATUS; return 1; }
  ( cd $REPO && $PY -W ignore $HARNESS/stage_score.py $out/pp --kind $kind ${truth:+--truth $truth} --metrics $out/metrics.json ) > $out/score.log 2>&1 || { echo FAILED_SCORE > $out/STATUS; return 1; }
  echo "OK $(( $(date +%s) - t0 ))s" > $out/STATUS
}
for ds in ${DATASETS//,/ }; do run_one $ds & done
wait

fail=$(grep -L '^OK' $RUN/*/STATUS 2>/dev/null | wc -l)
if [ "$fail" -eq 0 ]; then echo "DONE $(date -u +%FT%TZ)" > $RUN/STATUS; else echo "DONE_WITH_FAILURES $(date -u +%FT%TZ)" > $RUN/STATUS; fi
