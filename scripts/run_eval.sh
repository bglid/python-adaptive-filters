#!/usr/bin/bash

#---.sh file to run eval for all/any specific algorithm per noise type. Can be run from .sh, but easier command from Makefile---
set -euo pipefail

# checking that at least one positional argument was passed
if [ $# -lt 1 ]; then
  echo "Script usage: $0 <algorithm> [<algorithm> ...]"
  exit 1
fi

# passing args through entry point per noise type
# AC
python -m adaptive_filter.main \
  --filter_order=32 --mu=0.001 --algorithm "$@" --noise="air_conditioner" --save_result=True
