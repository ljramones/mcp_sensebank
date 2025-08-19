#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ~/venvs/strands/bin/activate
python -m sense_bank.agent "$@"
