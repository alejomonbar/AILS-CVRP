#!/bin/bash

# Example: Collecting baseline data with seeds for RL comparison
# This demonstrates how to use seeds for reproducible RL experiments

echo "====================================="
echo "Baseline Data Collection Example"
echo "====================================="
echo ""

# Test instance
INSTANCE="data/E-n22-k4.vrp"
BEST=375
TIME_LIMIT=30

# Output file
OUTPUT="baseline_results.csv"

echo "Instance,Seed,BestFound,Gap,Iterations,Time" > $OUTPUT

echo "Running baseline with 10 different seeds..."
echo ""

for seed in {1..10}; do
    echo "Running seed $seed..."
    
    # Run AILS-II with seed
    java -cp bin SearchMethod.AILSII \
        -file $INSTANCE \
        -rounded true \
        -best $BEST \
        -limit $TIME_LIMIT \
        -stoppingCriterion Time \
        -seed $seed \
        2>&1 | tee temp_run.txt
    
    # Extract metrics (you'll need to parse the output)
    # This is a placeholder - adjust based on actual output format
    BEST_FOUND=$(grep "solution quality:" temp_run.txt | tail -1 | awk '{print $3}')
    GAP=$(grep "gap:" temp_run.txt | tail -1 | awk '{print $5}')
    
    # Append to CSV
    echo "$INSTANCE,$seed,$BEST_FOUND,$GAP" >> $OUTPUT
    echo ""
done

echo "Results saved to $OUTPUT"
echo ""
echo "Next steps:"
echo "1. Analyze variance in baseline performance"
echo "2. Use same seeds for RL-enhanced version"
echo "3. Compare results using paired t-test"
echo ""

# Cleanup
rm -f temp_run.txt

echo "Example of comparing baseline vs RL:"
echo ""
echo "# Baseline"
echo "java -jar AILSII.jar -file data/E-n22-k4.vrp -seed 42 -limit 60"
echo ""
echo "# RL-enhanced (future implementation)"
echo "java -jar AILSII-RL.jar -file data/E-n22-k4.vrp -seed 42 -limit 60"
echo ""
echo "With same seed, any difference is due to RL decisions, not randomness!"
