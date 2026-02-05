#!/bin/bash

# Test script to verify reproducibility with random seed
# This script runs the same instance twice with the same seed
# and verifies the results are identical

echo "====================================="
echo "Reproducibility Test Script"
echo "====================================="
echo ""

# Test parameters
INSTANCE="data/E-n13-k4.vrp"
SEED=42
TIME_LIMIT=10
BEST=247

echo "Test 1: Running with seed=$SEED (first run)..."
java -cp bin SearchMethod.AILSII -file $INSTANCE -rounded true -best $BEST -limit $TIME_LIMIT -stoppingCriterion Time -seed $SEED > run1.txt

echo "Test 2: Running with seed=$SEED (second run)..."
java -cp bin SearchMethod.AILSII -file $INSTANCE -rounded true -best $BEST -limit $TIME_LIMIT -stoppingCriterion Time -seed $SEED > run2.txt

echo ""
echo "Comparing results..."
echo ""

# Extract solution qualities and compare
DIFF=$(diff run1.txt run2.txt)

if [ -z "$DIFF" ]; then
    echo "✅ SUCCESS: Both runs produced IDENTICAL output!"
    echo "   This confirms reproducibility with seed=$SEED"
else
    echo "❌ DIFFERENCE FOUND:"
    echo "$DIFF"
    echo ""
    echo "Note: Small timing differences are acceptable, but solution quality should be identical"
fi

echo ""
echo "Test 3: Running WITHOUT seed (should be different)..."
java -cp bin SearchMethod.AILSII -file $INSTANCE -rounded true -best $BEST -limit $TIME_LIMIT -stoppingCriterion Time > run3.txt

echo ""
echo "Comparing seeded run vs non-seeded run..."
DIFF2=$(diff run1.txt run3.txt)

if [ -z "$DIFF2" ]; then
    echo "⚠️  WARNING: Seeded and non-seeded runs produced same output"
    echo "   (This is very unlikely and suggests the seed isn't working)"
else
    echo "✅ Good: Seeded and non-seeded runs are different (as expected)"
fi

echo ""
echo "====================================="
echo "Test complete!"
echo "====================================="
echo ""
echo "Usage in your experiments:"
echo "  java -jar AILSII.jar -file <instance> -seed 42 -limit 100 -stoppingCriterion Time"
echo ""
echo "For RL experiments, use different seeds for:"
echo "  - Training: -seed 1, 2, 3, ..."
echo "  - Testing: -seed 1000, 1001, 1002, ..."
echo ""

# Cleanup
# rm run1.txt run2.txt run3.txt
