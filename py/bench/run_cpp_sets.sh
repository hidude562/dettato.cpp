#!/usr/bin/env bash
# The C++ pipeline (dettato_cli) on the four benchmark sets, then score.py.
#
#   py/bench/run_cpp_sets.sh SETS_DIR OUT_DIR MODEL.gguf [dettato_cli options, e.g. --no-drums]
#
# SETS_DIR is what build_sets.py wrote. The sets with drums (RWC-R here) are fed at their own rate, as a plugin would pass
# them (converted from full/<id>.flac to WAV with ffmpeg, cached in OUT_DIR/wav); the others get the set's 16 kHz audio,
# the same file the Python package reads. dettato_cli is taken from $BUILD (default: build) and score.py runs with $PYTHON.
set -euo pipefail
SETS=$1; OUT=$2; MODEL=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
CLI=${BUILD:-build}/tools/dettato_cli
PY=${PYTHON:-python3}
mkdir -p "$OUT"
for s in bach10 dcs phenicx rwc_r; do
  ins=()
  while read -r id; do
    if [ -f "$SETS/$s/full/$id.flac" ]; then
      w=$OUT/wav/$s/$id.wav
      [ -f "$w" ] || { mkdir -p "$(dirname "$w")"; ffmpeg -nostdin -v error -y -i "$SETS/$s/full/$id.flac" -c:a pcm_f32le "$w"; }
      ins+=("$w")
    else
      ins+=("$SETS/$s/wav/$id.wav")
    fi
  done < <("$PY" -c "import json,sys; [print(json.loads(l)['id']) for l in open(sys.argv[1])]" "$SETS/$s/manifest.jsonl")
  "$CLI" "$MODEL" "${ins[@]}" -o "$OUT/pipeline/$s" "$@"
done
"$PY" "$HERE/score.py" "$SETS" "$OUT/pipeline" --sets bach10,dcs,phenicx,rwc_r --out "$OUT/score" | tee "$OUT/score.txt"
