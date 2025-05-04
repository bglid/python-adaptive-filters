#!/usr/bin/bash

#---.sh file to run eval for all/any specific algorithm per noise type. Can be run from .sh---
set -euo pipefail

# checking that at least one positional argument was passed
if [ $# -lt 1 ]; then
  echo "Script usage: $0 <algorithm> [<algorithm> ...]"
  exit 1
fi

# passing args through entry point per noise type
python -m adaptive_filter.main \
  --eval=True --filter_order=32 --mu=0.001 --algorithm "$@" --noise="all"
