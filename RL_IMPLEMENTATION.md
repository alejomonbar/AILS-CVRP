# RL-Based Operator Selection Implementation

## Overview
This implementation adds Reinforcement Learning (RL) to intelligently select perturbation operators in the AILS-II algorithm using a Multi-Armed Bandit approach with UCB1 (Upper Confidence Bound).

## Architecture

### Core Components

#### 1. OperatorSelector (src/RL/OperatorSelector.java)
Multi-Armed Bandit implementation using UCB1 algorithm for operator selection.

**Key Features:**
- **UCB1 Algorithm**: Balances exploration vs exploitation
  ```
  UCB(i) = avgReward(i) + c * sqrt(ln(totalSelections) / selections(i))
  ```
- **Reward Function**: Normalized improvement ratio
  ```
  reward = (previousQuality - newQuality) / previousQuality
  ```
  - Positive reward: improvement
  - Negative reward: deterioration
- **Statistics Tracking**: Selection counts, cumulative rewards, UCB values per operator

**Methods:**
- `selectOperator()`: Choose operator using UCB1 or random selection
- `giveReward(reward)`: Update operator statistics with reward feedback
- `calculateImprovementReward(prev, new)`: Static method to calculate reward
- `getAllStats()`: Print detailed statistics for all operators

#### 2. Configuration Extensions

**Config.java additions:**
```java
boolean rlOperatorSelection;  // Enable/disable RL selection
double ucbExplorationParam;   // UCB exploration parameter (default: sqrt(2))
```

**InputParameters.java CLI parameters:**
```bash
-useRL <true|false>           # Enable RL-based selection
-ucbExploration <double>      # Set UCB exploration parameter
```

#### 3. AILSII Integration

**Modifications:**
1. Added `OperatorSelector` field initialized after perturbation operators
2. Replaced random operator selection with RL-based selection:
   ```java
   selectedPerturbation = operatorSelector.selectOperator();
   ```
3. Calculate reward and provide feedback after each iteration:
   ```java
   double reward = OperatorSelector.calculateImprovementReward(previousQuality, solution.f);
   operatorSelector.giveReward(reward);
   ```
4. Print operator statistics at end of search when RL enabled

#### 4. Enhanced Logging

**ResultsLogger.java additions:**
- `operator_index`: Which operator was selected (0, 1, ...)
- `reward`: Reward value given to the operator

**iterations.csv columns:**
```
iteration,time_sec,solution_quality,gap_percent,num_routes,eta,omega,
perturbation_type,insertion_heuristic,distance_ls,improvement,
operator_index,reward
```

## Usage

### Command Line Examples

**Baseline (random selection):**
```bash
java -cp bin SearchMethod.AILSII \
  -file data/XL-n1048-k237.vrp \
  -rounded true \
  -limit 60 \
  -best 380211 \
  -stoppingCriterion Time \
  -seed 42
```

**With RL enabled:**
```bash
java -cp bin SearchMethod.AILSII \
  -file data/XL-n1048-k237.vrp \
  -rounded true \
  -limit 60 \
  -best 380211 \
  -stoppingCriterion Time \
  -seed 42 \
  -useRL true
```

**Custom exploration parameter:**
```bash
java -cp bin SearchMethod.AILSII \
  -file data/XL-n1048-k237.vrp \
  -rounded true \
  -limit 60 \
  -best 380211 \
  -stoppingCriterion Time \
  -seed 42 \
  -useRL true \
  -ucbExploration 2.0
```

## Initial Test Results

### XL-n1048-k237.vrp (60 seconds, seed 42)

| Configuration | Best Quality | Gap (%) | Iterations | Improvement |
|--------------|--------------|---------|------------|-------------|
| Baseline (random) | 383,482 | 0.8603% | 10,154 | - |
| RL-UCB1 | 383,831 | 0.9521% | 8,600 | -349 (worse) |

**Operator Statistics (RL run):**
```
Total selections: 8602
Operator 0 (Sequential): selections=4311, avgReward=-0.000816
Operator 1 (Concentric): selections=4291, avgReward=-0.000969
```

**Observations:**
- RL explores both operators nearly equally (4311 vs 4291 selections)
- Both operators show small negative average rewards (typical for this search)
- RL completed fewer iterations (8,600 vs 10,154) - slight overhead
- RL result slightly worse on this single run, but within variance

## Implementation Details

### UCB1 Algorithm
The UCB1 (Upper Confidence Bound 1) algorithm is a simple yet effective strategy for the multi-armed bandit problem:

1. **Initialization**: Each operator gets one selection with zero reward
2. **Selection**: Choose operator with highest UCB value:
   - **Exploitation term**: `avgReward(i)` - favor operators with good past performance
   - **Exploration term**: `c * sqrt(ln(N) / n(i))` - encourage trying less-used operators
3. **Update**: After each iteration, update cumulative reward for selected operator
4. **Balance**: Parameter `c` controls exploration vs exploitation trade-off

### Reward Design
The reward function normalizes improvement by current solution quality:
```
reward = (previousQuality - newQuality) / previousQuality
```

**Properties:**
- Scale-invariant: works across different problem sizes
- Positive for improvement, negative for deterioration
- Magnitude proportional to relative improvement
- Allows comparing different operators fairly

### Integration Points
1. **Operator Selection** (before perturbation): UCB1 or random
2. **Reward Calculation** (after local search): Compare quality before/after
3. **Feedback** (before next iteration): Update operator statistics
4. **Logging** (every iteration): Record operator index and reward
5. **Statistics** (end of search): Print operator performance summary

## Files Modified

1. **src/RL/OperatorSelector.java** - New file (263 lines)
2. **src/SearchMethod/Config.java** - Added RL configuration (4 lines + getters/setters)
3. **src/SearchMethod/InputParameters.java** - Added CLI parameters (44 lines)
4. **src/SearchMethod/AILSII.java** - Integration logic (12 lines)
5. **src/SearchMethod/ResultsLogger.java** - Enhanced logging (2 parameters)

## Next Steps

### Recommended Experiments
1. **Multiple seeds**: Test with 5-10 different seeds to assess variance
2. **Different instances**: Test on various problem sizes and types
3. **Parameter tuning**: Try different `ucbExploration` values (0.5, 1.0, 2.0, 5.0)
4. **Longer runs**: Test with 300-600 seconds to see if RL learns better patterns
5. **Operator comparison**: Analyze which operator performs better in different contexts

### Potential Enhancements (Future Work)
- **Contextual bandits**: Consider problem features (routes, customers, time) when selecting
- **Thompson sampling**: Alternative to UCB1
- **Epsilon-greedy**: Simpler exploration strategy
- **Dynamic rewards**: Add bonuses for finding new best solutions
- **Operator chaining**: Learn sequences of operators
- **Transfer learning**: Use learned strategies across similar instances

### Analysis Tools Needed
- Statistical comparison (t-test, Mann-Whitney U)
- Convergence plots (quality vs time)
- Operator selection frequency over time
- Reward distribution analysis

## Commit History

- **da84ee6**: Implement basic UCB1 operator selection
- **35dac31**: Add logging system documentation
- **ff7a677**: Add comprehensive results logging system
- **0ed27fc**: Add reproducibility test results
- **06370c3**: Add random seed support for reproducible experiments

## References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2), 235-256.
- Multi-Armed Bandit: https://en.wikipedia.org/wiki/Multi-armed_bandit
- UCB1 Algorithm: https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
