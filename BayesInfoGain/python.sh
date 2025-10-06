#!/bin/bash

# Ensure a script was provided
if [ -z "$1" ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi

SCRIPT_PATH="$1"
# SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
BASE_DIR=$(pwd)

# Run the Python script with PYTHONPATH set
PYTHONPATH="$BASE_DIR" python "$SCRIPT_PATH"
