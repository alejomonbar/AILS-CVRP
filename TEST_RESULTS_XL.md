# Test Results: Reproducibility with XL-n1048-k237

## Instance Details
- **File**: data/XL-n1048-k237.vrp
- **Best Known Solution (BKS)**: 380,211
- **Problem size**: 1,048 customers, 237 vehicles

## Seed Implementation: ✅ VERIFIED WORKING

### Test 1: Iteration-Based Stopping (Perfect Reproducibility)
```bash
java -cp bin SearchMethod.AILSII \
  -file data/XL-n1048-k237.vrp \
  -rounded true \
  -best 380211 \
  -limit 100 \
  -stoppingCriterion Iteration \
  -seed 42
```

**Results (Run 1 vs Run 2)**:
| Iteration | Solution Quality | Gap | K | Eta | Omega | Match? |
|-----------|-----------------|-----|---|-----|-------|--------|
| 1 | 398047.0 | 4.6911% | 250 | 1.0000 | 30.0000 | ✅ |
| 90 | 390869.0 | 2.8032% | 246 | 0.0631 | 4.6948 | ✅ |
| 91 | 390730.0 | 2.7666% | 246 | 0.0603 | 3.8455 | ✅ |
| 97 | 389548.0 | 2.4557% | 245 | 0.0457 | 4.6948 | ✅ |

**Conclusion**: With same seed, every iteration produces **identical** solution quality, gap, number of routes, eta, and omega values.

### Test 2: Time-Based Stopping (Decisions Deterministic, Timing Variable)
```bash
java -cp bin SearchMethod.AILSII \
  -file data/XL-n1048-k237.vrp \
  -rounded true \
  -best 380211 \
  -limit 10 \
  -stoppingCriterion Time \
  -seed 42
```

**Observations**:
- First iteration is always identical: 398047.0 (seed 42)
- Each iteration produces same decisions
- But different system load → different number of iterations completed in 10 seconds
- **Not an issue**: Decisions are deterministic, just # of iterations varies

### Test 3: Different Seeds Produce Different Results
```bash
# Seed 42
solution quality: 398047.0 gap: 4.6911% K: 250 iteration: 1

# Seed 99  
solution quality: 398224.0 gap: 4.7376% K: 247 iteration: 1
```

✅ Different seeds → different solutions (as expected)

## Recommendations for RL Experiments

### 1. Use Iteration-Based Stopping for Training
```bash
# Baseline data collection
for seed in {1..30}; do
    java -cp bin SearchMethod.AILSII \
        -file data/XL-n1048-k237.vrp \
        -rounded true \
        -best 380211 \
        -limit 1000 \
        -stoppingCriterion Iteration \
        -seed $seed
done
```

**Why?**: Perfect reproducibility, no timing variance.

### 2. Use Time-Based for Final Comparison
```bash
# Real-world scenario: 60 seconds budget
java -cp bin SearchMethod.AILSII \
    -file data/XL-n1048-k237.vrp \
    -seed 42 \
    -limit 60 \
    -stoppingCriterion Time
```

**Why?**: Reflects actual deployment where time matters.

### 3. Seed Strategy for RL Development

**Training Set**: Seeds 1-1000
```bash
# Train RL agent on diverse random seeds
for seed in {1..1000}; do
    # Run and collect decision data
done
```

**Validation Set**: Seeds 10001-10100
```bash
# Check overfitting during training
```

**Test Set**: Seeds 20001-20100
```bash
# Final evaluation on completely unseen seeds
```

### 4. Comparing Baseline vs RL-Enhanced

**Paired Comparison** (same seed for both):
```bash
# Baseline
java -jar AILSII.jar -file instance.vrp -seed 42 -limit 1000 -stoppingCriterion Iteration

# RL-enhanced (future)
java -jar AILSII-RL.jar -file instance.vrp -seed 42 -limit 1000 -stoppingCriterion Iteration

# Any difference is purely from RL decisions!
```

## Performance Baseline (60 seconds, seed 12345)

**XL-n1048-k237** with seed 12345:
- Starting solution: 398674.0 (gap 4.86%)
- After 60s: 382947.0 (gap 0.72%)
- Final K: 241 routes (vs 237 minimum)
- Improvement: 15727 units (3.9% reduction)
- Iterations: ~9669 in 60 seconds

**Key Metrics for RL to Beat**:
- Gap after 100 iterations
- Gap after 1000 iterations  
- Final gap after fixed iterations
- Time to reach 1% gap
- Time to reach 0.5% gap

## Seed Implementation Status

✅ **All Random objects seeded**:
- AILSII main search ✅
- Perturbation (Sequential & Concentric) ✅
- OmegaAdjustment ✅
- ConstructSolution ✅
- AcceptanceCriterion (no Random) ✅

✅ **Parameter added**: `-seed <long>`

✅ **Backward compatible**: No seed = original non-deterministic behavior

✅ **Documented**: README.md updated

✅ **Tested**: Reproducibility verified

## Next Actions

1. ✅ **Reproducibility verified** - Ready for RL experiments
2. **Collect baseline data** - Run with multiple seeds to understand variance
3. **Implement first RL component** - Start with operator selection bandit
4. **Compare with paired seeds** - Measure RL impact precisely

---

**Test Date**: February 5, 2026  
**Branch**: rl-improvement  
**Status**: ✅ READY FOR RL DEVELOPMENT
