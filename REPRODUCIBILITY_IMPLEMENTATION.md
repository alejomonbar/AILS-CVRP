# Random Seed Implementation for Reproducibility

## Summary
Added random seed support to the AILS-II codebase to enable reproducible experiments. This is essential for comparing RL-enhanced versions against the baseline.

## Changes Made

### 1. Configuration (`Config.java`)
- Added `Long randomSeed` field (null = non-deterministic, otherwise use specified seed)
- Added getter/setter methods
- Default: `null` (maintains original non-deterministic behavior)

### 2. Input Parameters (`InputParameters.java`)
- Added `-seed` command-line parameter handling
- Added `getSeed()` method to parse seed value
- Validates input is a valid long integer

### 3. Random Number Generator Initialization
Modified all classes that use `Random`:

#### `AILSII.java` (main search)
```java
if(config.getRandomSeed() != null) {
    this.rand = new Random(config.getRandomSeed());
} else {
    this.rand = new Random();
}
```

#### `Perturbation.java` (perturbation operators)
- Both Sequential and Concentric inherit this seeded Random

#### `OmegaAdjustment.java` (diversity control)
- Omega parameter adjustments now reproducible

#### `ConstructSolution.java` (initial solution)
- Initial solution construction now reproducible

## Usage

### With Seed (Reproducible)
```bash
java -jar AILSII.jar -file data/E-n13-k4.vrp -rounded true -best 247 -limit 100 -stoppingCriterion Time -seed 42
```

### Without Seed (Original Behavior)
```bash
java -jar AILSII.jar -file data/E-n13-k4.vrp -rounded true -best 247 -limit 100 -stoppingCriterion Time
```

## Testing

Run the reproducibility test script:
```bash
./test_reproducibility.sh
```

This script:
1. Runs the same instance twice with the same seed
2. Verifies outputs are identical
3. Runs without seed and confirms different behavior

## Why This Matters for RL

### Baseline Comparison
When testing RL improvements, you can now:
```bash
# Baseline run with seed
java -jar AILSII.jar -file instance.vrp -seed 42 -limit 100 > baseline.txt

# RL-enhanced run with same seed
java -jar AILSII-RL.jar -file instance.vrp -seed 42 -limit 100 > rl_version.txt

# Now differences are due to RL changes, not random variation
```

### Statistical Testing
```bash
# Run multiple trials with different seeds
for seed in {1..30}; do
    java -jar AILSII.jar -file instance.vrp -seed $seed -limit 100 >> results.csv
done
```

### Debugging
When RL makes a bad decision, you can:
1. Note the seed that caused the issue
2. Rerun with same seed to reproduce exact behavior
3. Debug step-by-step with deterministic execution

## Implementation Notes

### Seed Propagation
**Problem**: Multiple `Random` objects in different classes
**Solution**: All classes receive `Config` and check `getRandomSeed()`

### Null vs 0
**Important**: We use `Long` (not `long`) so `null` means "no seed"
- `null` → Non-deterministic (original behavior)
- `0` → Valid seed that produces deterministic sequence

### Same Seed Everywhere
**Design Decision**: All Random objects use the SAME seed
- **Pro**: Simpler implementation, full reproducibility
- **Con**: Not independent random streams
- **Alternative** (if needed): Could use `seed + offset` for each class

## Validation

To verify the implementation works:

```bash
# Run 1
java -jar AILSII.jar -file data/E-n13-k4.vrp -seed 12345 -limit 10 > out1.txt

# Run 2  
java -jar AILSII.jar -file data/E-n13-k4.vrp -seed 12345 -limit 10 > out2.txt

# Should be identical
diff out1.txt out2.txt
```

## Next Steps for RL Development

1. **Data Collection**: Run baseline with multiple seeds to understand variance
   ```bash
   for seed in {1..100}; do
       java -jar AILSII.jar -file data/E-n51-k5.vrp -seed $seed -limit 60 >> baseline_stats.csv
   done
   ```

2. **RL Training**: Use seeds 1-1000 for training instances

3. **RL Testing**: Use seeds 10000-10100 for testing (completely separate)

4. **Comparison**: Same seed for baseline vs RL to isolate RL impact
   ```bash
   # Baseline
   java -jar AILSII.jar -file test.vrp -seed 42 > baseline_42.txt
   
   # RL version
   java -jar AILSII-RL.jar -file test.vrp -seed 42 > rl_42.txt
   
   # Compare decisions at each iteration
   ```

## Files Modified

- `src/SearchMethod/Config.java` - Added seed field
- `src/SearchMethod/InputParameters.java` - Added seed parameter
- `src/SearchMethod/AILSII.java` - Seeded main Random
- `src/Perturbation/Perturbation.java` - Seeded perturbation Random
- `src/DiversityControl/OmegaAdjustment.java` - Seeded omega Random
- `src/SearchMethod/ConstructSolution.java` - Seeded construction Random
- `README.md` - Added seed documentation
- `test_reproducibility.sh` - New test script

## Backward Compatibility

✅ **Fully backward compatible**
- Existing scripts without `-seed` work exactly as before
- Default behavior unchanged (non-deterministic)
- Only opt-in when seed is specified

---

**Date**: February 5, 2026  
**Branch**: rl-improvement  
**Status**: Implemented and tested
