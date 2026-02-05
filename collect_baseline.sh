#!/bin/bash

# Script to run multiple experiments and collect baseline data

INSTANCE="data/XL-n1048-k237.vrp"
BEST=380211
ITERATIONS=1000

echo "======================================"
echo "Baseline Data Collection"
echo "======================================"
echo ""
echo "Instance: $INSTANCE"
echo "Iterations per run: $ITERATIONS"
echo ""

# Create experiments directory
mkdir -p experiments

# Run with multiple seeds
for seed in 1 2 3 5 7 11 13 17 19 23; do
    echo "Running seed $seed..."
    java -cp bin SearchMethod.AILSII \
        -file $INSTANCE \
        -rounded true \
        -best $BEST \
        -limit $ITERATIONS \
        -stoppingCriterion Iteration \
        -seed $seed
    echo ""
done

echo "======================================"
echo "Baseline collection complete!"
echo "======================================"
echo ""
echo "Results saved in: experiments/XL-n1048-k237/"
echo ""
echo "To analyze results, run:"
echo "  python analyze_results.py experiments/XL-n1048-k237"
