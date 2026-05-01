#!/bin/bash
source ~/.bashrc
conda activate cot
cd "$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="src:${PYTHONPATH:-}"
python -c "$1"
