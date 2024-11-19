#!/bin/bash

# Experiment A (all on all): 10 rep x 6 data x 4 model = 240
for i in $(seq 0 239); do
    python david.py --setting $i
    echo
    echo
done