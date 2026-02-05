# RL Decision Points in AILS-II Code - Analysis

## Overview
Analysis of the existing AILS-II Java codebase to identify specific decision points where Reinforcement Learning could replace or enhance heuristic rules.

---

## 1. **PERTURBATION OPERATOR SELECTION** 🎯 HIGH IMPACT
**Location**: [SearchMethod/AILSII.java:138](src/SearchMethod/AILSII.java#L138)

### Current Implementation:
```java
selectedPerturbation=pertubOperators[rand.nextInt(pertubOperators.length)];
```

### Problem:
- **Random uniform selection** between Concentric and Sequential perturbations
- No learning from which operator works better in different situations
- Misses patterns about when each operator is effective

### RL Opportunity:
**Multi-Armed Bandit or Contextual Bandit**

**State Features**:
- Solution quality (gap from BKS)
- Iterations since last improvement
- Current eta value
- Current omega value  
- Average route length
- Number of routes vs minimum possible
- Problem instance characteristics (size, density, capacity ratio)

**Actions**: {Concentric, Sequential}

**Reward**: Immediate solution improvement after perturbation + local search

**Expected Gain**: 
- Learn instance-specific patterns (e.g., Sequential better for clustered customers)
- Adapt strategy during search (e.g., more aggressive early, conservative late)
- 5-15% better operator selection efficiency

---

## 2. **INSERTION HEURISTIC SELECTION** 🎯 HIGH IMPACT
**Location**: [Perturbation/Perturbation.java:94](src/Perturbation/Perturbation.java#L94)

### Current Implementation:
```java
indexHeuristic=rand.nextInt(insertionHeuristics.length);
selectedInsertionHeuristic=insertionHeuristics[indexHeuristic];
```

### Problem:
- Random selection between Distance-based and Cost-based insertion
- No adaptation based on which works better

### RL Opportunity:
**Contextual Bandit**

**State Features**:
- Number of nodes being reinserted (omega value)
- Perturbation type used
- Current solution structure
- Customer distribution patterns

**Actions**: {InsertionHeuristic.Distance, InsertionHeuristic.Cost}

**Reward**: Quality of solution after reinsertion

**Expected Gain**: 10-20% improvement in reinsertion quality

---

## 3. **OMEGA (PERTURBATION STRENGTH) ADJUSTMENT** 🎯 VERY HIGH IMPACT
**Location**: [DiversityControl/OmegaAdjustment.java:45-51](src/DiversityControl/OmegaAdjustment.java#L45-L51)

### Current Implementation:
```java
public void setupOmega() {
    obtainedDist=meanLSDist.getDynamicAverage();
    omega+=((omega/obtainedDist*idealDist.idealDist)-omega);
    omega=Math.min(omegaMax, Math.max(omega, omegaMin));
    averageOmega.setValue(omega);
    iterator=0;
}
```

### Problem:
- Fixed formula: `omega += ((omega/obtainedDist*idealDist.idealDist) - omega)`
- Simple proportional adjustment doesn't account for:
  - Search phase (early exploration vs late intensification)
  - Problem characteristics
  - Recent success/failure patterns
  - Time pressure

### RL Opportunity:
**Policy Gradient / Actor-Critic**

**State Features**:
- Current omega value
- Mean distance from local search (obtainedDist)
- Ideal distance target (idealDist.idealDist)
- Improvement history (moving average)
- Time/iteration budget remaining
- Stagnation indicator (iterations without improvement)
- Solution quality progression

**Actions**: 
- Continuous: omega ∈ [1, size-2]
- Or discrete: {decrease_large, decrease_small, keep, increase_small, increase_large}

**Reward**:
```
R = α₁ * quality_improvement 
  + α₂ * diversity_maintained
  - α₃ * time_cost
```

**Expected Gain**: 15-25% better convergence, avoid premature convergence or excessive exploration

---

## 4. **ETA (ACCEPTANCE CRITERION) ADJUSTMENT** 🎯 HIGH IMPACT
**Location**: [DiversityControl/AcceptanceCriterion.java:72-79](src/DiversityControl/AcceptanceCriterion.java#L72-L79)

### Current Implementation:
```java
alpha=Math.pow(etaMin/etaMax, (double) 1/executionMaximumLimit);
eta*=alpha;
eta=Math.max(eta, etaMin);
```

### Problem:
- Fixed exponential decay schedule
- Same schedule regardless of search progress
- Doesn't react to solution quality or stagnation

### RL Opportunity:
**Adaptive Policy Learning**

**State Features**:
- Current eta value
- Best solution found so far
- Average solution quality (averageLSfunction)
- Upper limit threshold
- Search progress (time/iteration percentage)
- Improvement rate

**Actions**: Adjust eta rate (faster/slower decay, or pause decay)

**Reward**: Balance between:
- Accepting improving moves (good)
- Accepting diverse moves when stuck (good)
- Rejecting too many moves (bad - waste computation)

**Expected Gain**: 10-15% better exploration-exploitation balance

---

## 5. **REFERENCE NODE SELECTION IN CONCENTRIC** 🎯 MEDIUM IMPACT
**Location**: [Perturbation/Concentric.java:27](src/Perturbation/Concentric.java#L27)

### Current Implementation:
```java
Node reference=solution[rand.nextInt(solution.length)];
```

### Problem:
- Random selection of reference node
- Some nodes might be better perturbation centers than others
- Doesn't consider strategic locations (cluster centers, route boundaries)

### RL Opportunity:
**Value Function Learning**

**State**: Current solution structure

**Actions**: Select reference node from solution

**Node Features for Selection**:
- Node degree (number of nearby customers)
- Position in route (start, middle, end)
- Demand of node
- Distance from depot
- Clustering coefficient
- Historical perturbation success starting from similar nodes

**Reward**: Quality after perturbation

**Expected Gain**: 5-10% better perturbations

---

## 6. **REINSERTION ORDER (setOrder)** 🎯 MEDIUM IMPACT  
**Location**: [Perturbation/Perturbation.java:65-73](src/Perturbation/Perturbation.java#L65-L73)

### Current Implementation:
```java
public void setOrder() {
    for (int i = 0; i < countCandidates; i++) {
        indexA=rand.nextInt(countCandidates);
        indexB=rand.nextInt(countCandidates);
        aux=candidates[indexA];
        candidates[indexA]=candidates[indexB];
        candidates[indexB]=aux;
    }
}
```

### Problem:
- Random shuffling of removed nodes before reinsertion
- Order matters! Some sequences lead to better solutions
- No exploitation of problem structure

### RL Opportunity:
**Sequence Learning (Pointer Network / Transformer)**

**Input**: Set of removed nodes with features
- Node coordinates
- Demand
- Original position/route
- KNN relationships

**Output**: Ordered sequence for reinsertion

**Training**: Learn from successful reinsertions in historical data

**Expected Gain**: 8-15% better reinsertion quality

---

## 7. **LOCAL SEARCH MOVE SELECTION** 🎯 MEDIUM-HIGH IMPACT
**Location**: [Improvement/LocalSearch.java:135-141](src/Improvement/LocalSearch.java#L135-L141)

### Current Implementation:
```java
public void browseRoutes(Route route) {
    if(route.numElements>1) {
        searchBestSHIFT(route);
        searchBestSwapStarKnn(route);
        searchBestCross(route);
    }
}
```

### Problem:
- Fixed order: SHIFT → SwapStar → Cross
- All moves evaluated exhaustively over KNN
- Computational waste on unpromising moves

### RL Opportunity:
**Q-Learning for Move Prioritization**

**State Features**:
- Route characteristics (length, slack capacity, node density)
- Previous move success patterns
- Search progress

**Actions**: {SHIFT, SwapStar, Cross} and intensity level

**Q-Value**: Expected improvement from each move type

**Reward**: Actual improvement found

**Expected Gain**: 20-30% speedup by skipping unpromising neighborhoods, same or better quality

---

## 8. **KNN LIMIT IN LOCAL SEARCH** 🎯 MEDIUM IMPACT
**Location**: [SearchMethod/Config.java](src/SearchMethod/Config.java) (varphi parameter)

### Current Problem:
```java
this.limitAdj=Math.min(config.getVarphi(), instance.getSize()-1);
```

- Fixed limit (default 40) for all instances and all search phases
- Computational vs quality tradeoff

### RL Opportunity:
**Adaptive Neighborhood Sizing**

**State**: Search progress, solution quality, time remaining

**Actions**: Adjust limitAdj ∈ [5, 100]

**Strategy**:
- Early search: Smaller neighborhoods (faster, more iterations)
- Late search: Larger neighborhoods (thorough, final polish)
- When stuck: Expand neighborhood

**Expected Gain**: 15-20% speedup with maintained quality

---

## 9. **SEQUENTIAL PERTURBATION STRING LENGTH** 🎯 LOW-MEDIUM IMPACT
**Location**: [Perturbation/Sequential.java:32](src/Perturbation/Sequential.java#L32)

### Current Implementation:
```java
sizeString=Math.min(Math.max(1, size),(int)omega-countCandidates);
```

### Problem:
- Length determined by remaining omega budget
- Could be smarter about string lengths

### RL Opportunity:
**Learn optimal string length distribution**

**State**: Current route structure, omega value

**Actions**: Choose string length for next removal

**Expected Gain**: 3-7% improvement in perturbation quality

---

## 10. **INITIAL SOLUTION CONSTRUCTION** 🎯 LOW IMPACT (but interesting)
**Location**: [SearchMethod/ConstructSolution.java](src/SearchMethod/ConstructSolution.java)

### RL Opportunity:
**Graph Neural Network for Initial Construction**

Replace heuristic construction with learned policy:
- GNN encodes problem instance
- Decoder autoregressively builds routes
- Trained on good solutions

**Expected Gain**: Better starting point → 5-10% faster convergence

---

## Priority Ranking for Implementation

### Tier 1 - Highest ROI (implement first):
1. **Omega adjustment** (RL replaces formula) - Core parameter, high impact
2. **Perturbation operator selection** - Simple bandit, quick win
3. **Local search move prioritization** - Big speedup potential

### Tier 2 - Good Returns:
4. **Eta adjustment** - Improves acceptance strategy
5. **Insertion heuristic selection** - Clear decision point
6. **KNN limit adaptation** - Performance/quality tradeoff

### Tier 3 - Incremental:
7. **Reference node selection** - Marginal but interesting
8. **Reinsertion order** - Complex, needs sequence model
9. **String length in Sequential** - Minor impact
10. **Initial construction** - Nice to have, separate research direction

---

## Recommended Starting Point

### **Quick Win: Multi-Armed Bandit for Operator Selection**

**Why start here?**
- ✅ Minimal code changes
- ✅ Clear reward signal
- ✅ Fast to implement and test
- ✅ Proves RL value before bigger investments
- ✅ No deep learning needed

**Implementation Sketch** (Java):
```java
public class BanditOperatorSelector {
    private double[] successCounts;  // Successes per operator
    private double[] trialCounts;    // Total trials per operator
    private double explorationParam; // UCB exploration coefficient
    
    public int selectOperator(SearchState state) {
        // UCB1 algorithm
        double[] ucbValues = new double[numOperators];
        for (int i = 0; i < numOperators; i++) {
            double exploitation = successCounts[i] / trialCounts[i];
            double exploration = Math.sqrt(
                explorationParam * Math.log(totalTrials) / trialCounts[i]
            );
            ucbValues[i] = exploitation + exploration;
        }
        return argmax(ucbValues);
    }
    
    public void updateReward(int operator, double improvement) {
        trialCounts[operator]++;
        if (improvement > 0) {
            successCounts[operator] += improvement; // or binary: +1
        }
    }
}
```

**Expected Results**:
- 2-3 days implementation
- Clear improvement metrics
- Foundation for more complex RL

---

## Integration Architecture

### Option A: Minimal Java-only
```
AILSII
  └─> DecisionManager (Java)
        ├─> BanditOperatorSelector
        ├─> AdaptiveOmegaController  
        └─> LocalSearchPrioritizer
```

### Option B: Hybrid (for Deep RL later)
```
AILSII (Java)
  └─> RLInterface
        ├─> Simple decisions (Java Bandits)
        └─> Complex policies (Python via socket/REST)
              └─> PyTorch models
```

---

## Measurement Framework

### What to Log:
```java
public class RLDecisionLog {
    long timestamp;
    int iteration;
    String decisionType;  // "perturbation", "omega", "eta", etc.
    double[] state;       // State features
    int action;           // Action taken
    double reward;        // Reward received
    double solutionQuality;
    
    public void log() {
        // Write to CSV or JSON for analysis
    }
}
```

### Metrics to Track:
- **Regret**: Gap vs optimal decision
- **Improvement rate**: Solutions improving per iteration
- **Sample efficiency**: Learning speed
- **Generalization**: Performance on unseen instances

---

## Next Steps

1. **Instrument current code** - Add logging at decision points
2. **Collect baseline data** - Run AILS-II with extensive logging
3. **Analyze patterns** - Which decisions correlate with improvement?
4. **Implement Tier 1 #2** - Bandit operator selection (quick win)
5. **Evaluate and iterate** - Measure impact, refine approach
6. **Scale to Tier 1 #1 & #3** - More complex RL components

---

**Author**: Analysis of AILS-II codebase  
**Date**: February 5, 2026  
**Branch**: rl-improvement
