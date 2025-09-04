#!/bin/bash

if [ -z "$1" ]; then
  echo "Please provide a folder"
  exit 1
fi

DIR=$1

python evaluate_docci500/eval_lf.py $DIR

python evaluate_docci500/eval_lf_turn2.py $DIR

python evaluate_docci500/eval_CAPTURE_lf.py $DIR

python evaluate_docci500/eval_CAPTURE_lf_turn2.py $DIR
