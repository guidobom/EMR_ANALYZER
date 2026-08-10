#!/bin/bash
# EMR Analyzer — Launch wrapper using the conda environment's Python directly

"$HOME/miniconda3/envs/emr-analyzer/bin/python" "$(dirname "$0")/run.py"
