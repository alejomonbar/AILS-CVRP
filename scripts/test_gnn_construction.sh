#!/bin/bash

# Test GNN Construction Heuristic Integration
# This script tests the GNN-based initial solution construction

echo "======================================"
echo "GNN Construction Heuristic Test"
echo "======================================"
echo ""

# Configuration
INSTANCE="data/E-n101-k14.vrp"
TIME_LIMIT=60
SEEDS=(42 100 200)

# Check if instance exists
if [ ! -f "$INSTANCE" ]; then
    echo "Error: Instance file not found: $INSTANCE"
    exit 1
fi

# Check if model exists
if [ ! -f "models/gnn_construction.pt" ]; then
    echo "Warning: GNN model not found at models/gnn_construction.pt"
    echo "Please train the model first using:"
    echo "  cd python_ml"
    echo "  python generate_training_data.py"
    echo "  python train_gnn.py --epochs 100"
    echo "  python export_model.py --checkpoint checkpoints/best_model.pt --output ../models/gnn_construction.pt"
    echo ""
    echo "Skipping GNN test. Running baseline only."
    TEST_GNN=false
else
    TEST_GNN=true
fi

# Create results directory
mkdir -p results/gnn_test

echo "Test Configuration:"
echo "  Instance: $INSTANCE"
echo "  Time limit: ${TIME_LIMIT}s"
echo "  Seeds: ${SEEDS[@]}"
echo "  GNN available: $TEST_GNN"
echo ""

# Function to run a single test
run_test() {
    local use_gnn=$1
    local seed=$2
    local label=$3
    
    echo "----------------------------------------"
    echo "Running: $label (seed=$seed)"
    echo "----------------------------------------"
    
    if [ "$use_gnn" = true ]; then
        mvn exec:java -Dexec.mainClass="Main.Main" -Dexec.args="\
            -i $INSTANCE \
            -t $TIME_LIMIT \
            -s $seed \
            -useGNN true \
            -gnnModelPath models/gnn_construction.pt" 2>&1 | tee "results/gnn_test/${label}_seed${seed}.log"
    else
        mvn exec:java -Dexec.mainClass="Main.Main" -Dexec.args="\
            -i $INSTANCE \
            -t $TIME_LIMIT \
            -s $seed" 2>&1 | tee "results/gnn_test/${label}_seed${seed}.log"
    fi
    
    echo ""
}

# Run baseline tests (classical construction)
echo "======================================"
echo "Phase 1: Baseline (Classical Construction)"
echo "======================================"
echo ""

for seed in "${SEEDS[@]}"; do
    run_test false $seed "baseline"
done

# Run GNN tests if model is available
if [ "$TEST_GNN" = true ]; then
    echo ""
    echo "======================================"
    echo "Phase 2: GNN Construction"
    echo "======================================"
    echo ""
    
    for seed in "${SEEDS[@]}"; do
        run_test true $seed "gnn"
    done
fi

# Extract and compare results
echo ""
echo "======================================"
echo "Results Summary"
echo "======================================"
echo ""

echo "Extracting results from logs..."
echo ""

printf "%-20s %-10s %-15s %-15s\n" "Configuration" "Seed" "Final Cost" "Gap %"
printf "%-20s %-10s %-15s %-15s\n" "--------------------" "----------" "---------------" "---------------"

# Extract baseline results
for seed in "${SEEDS[@]}"; do
    log_file="results/gnn_test/baseline_seed${seed}.log"
    if [ -f "$log_file" ]; then
        cost=$(grep -E "Best solution|Final cost" "$log_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+|[0-9]+' | head -1)
        gap=$(grep -E "Gap|gap" "$log_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        printf "%-20s %-10s %-15s %-15s\n" "Baseline" "$seed" "${cost:-N/A}" "${gap:-N/A}"
    fi
done

# Extract GNN results if available
if [ "$TEST_GNN" = true ]; then
    for seed in "${SEEDS[@]}"; do
        log_file="results/gnn_test/gnn_seed${seed}.log"
        if [ -f "$log_file" ]; then
            cost=$(grep -E "Best solution|Final cost" "$log_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+|[0-9]+' | head -1)
            gap=$(grep -E "Gap|gap" "$log_file" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            printf "%-20s %-10s %-15s %-15s\n" "GNN" "$seed" "${cost:-N/A}" "${gap:-N/A}"
        fi
    done
fi

echo ""
echo "======================================"
echo "Test completed!"
echo "Results saved in: results/gnn_test/"
echo "======================================"
