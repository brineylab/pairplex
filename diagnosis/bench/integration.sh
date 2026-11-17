#!/bin/bash
# Post-merge integration check for origin/main (run after every merge into main).
#   integration.sh [REF]        default REF = origin/main
# 1. Snapshots REF (git archive) and runs the FULL test suite (pairplex + simplex) on this pod,
#    within the agent limit (2 workers, 1 thread each).
# 2. Queues one benchmark of REF through the normal 2-slot queue (name main-<sha7>).
# 3. Writes ~/work/PairPlex/bench/integration/<sha7>.md. When the benchmark is DONE, run:
#      python diagnosis/bench/compare.py main-<sha7> --baseline <previous main run or v2-base>
set -uo pipefail
REPO=/home/jovyan/work/Software/pairplex
PY=/home/jovyan/work/pairplex-simplex/.venv/bin/python
OUT=/home/jovyan/work/PairPlex/bench/integration
REF=${1:-origin/main}
git -C $REPO fetch -q origin
SHA=$(git -C $REPO rev-parse --verify "$REF^{commit}") || { echo "unknown ref $REF"; exit 1; }
S7=${SHA:0:7}
SRC=$OUT/src_$S7
mkdir -p $OUT
if [ ! -d "$SRC" ]; then
  mkdir -p "$SRC.tmp" && git -C $REPO archive $SHA | tar --no-same-owner -x -C "$SRC.tmp" && mv "$SRC.tmp" "$SRC"
fi
REC=$OUT/$S7.md
{
  echo "# Integration check: $REF @ $SHA"
  echo "- started: $(date -u +%FT%TZ)"
  echo "- merges since v2-base:"; git -C $REPO log --merges --format='  - %h %s' v2-base..$SHA
} > $REC

# 1. full test suite (local, limited)
cd "$SRC"
PAIRPLEX_MAX_WORKERS=2 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 timeout 3600 \
  $PY -m pytest -q -p no:cacheprovider pairplex simplex > $OUT/$S7.pytest.log 2>&1
rc=$?
echo "- tests: $(tail -1 $OUT/$S7.pytest.log) (exit $rc; log $S7.pytest.log)" >> $REC

# 2. benchmark (queued; respects the 2-slot limit)
NAME=main-$S7
if [ -e /home/jovyan/work/PairPlex/bench/runs/$NAME ]; then
  echo "- benchmark: $NAME already exists" >> $REC
else
  bash $REPO/diagnosis/bench/submit.sh $SHA $NAME '{}' >> $REC 2>&1
  echo "- benchmark: queued as $NAME (default kwargs)" >> $REC
fi
echo "- next: compare.py $NAME --baseline <previous main-* run or v2-base>; neutral merges must be SAME" >> $REC
cat $REC
exit $rc
