#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <desired_accuracy>"
    exit 1
fi

while true; do
    python3 model/K-fold_hyperparameter_optimization.py
done
