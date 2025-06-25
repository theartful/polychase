#!/bin/sh

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

source "$SCRIPT_DIR/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/build/cpp"

cd "$SCRIPT_DIR/build"
pybind11-stubgen polychase_core --numpy-array-use-type-var && cp ./stubs/polychase_core.pyi "$SCRIPT_DIR/blender_addon/lib/polychase_core.pyi"
