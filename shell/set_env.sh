#! /bin/bash

PYTHON_BIN_PATH=`which python`
export PYTHON_BIN_DIR=${PYTHON_BIN_PATH%/*}
export WORK_DIR=`pwd`
export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"
export PATH="$PYTHON_BIN_DIR:$PATH"

mkdir -p data
mkdir -p model
mkdir -p logs
