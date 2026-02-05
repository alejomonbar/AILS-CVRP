# Reinforcement Learning Improvements for AILS-II CVRP Solver

## Overview
This document outlines a comprehensive plan to integrate Reinforcement Learning (RL) techniques into the existing AILS-II (Adaptive Iterated Local Search) framework for the Capacitated Vehicle Routing Problem (CVRP).

## Current Architecture Analysis

### Existing AILS-II Components
1. **Adaptive Parameters**: 
   - `omega` (perturbation strength) - adjusted based on distance metrics
   - `eta` (acceptance criterion threshold) - controls solution acceptance
   - `dMin/dMax` (ideal distance bounds)

2. **Main Algorithm Flow**:
   ```
   Construct initial solution → Local Search
   While (stopping criterion not met):
      - Apply perturbation (strength ω)
      - Make feasible
      - Local search
      - Calculate distance
      - Accept/reject based on η
      - Adjust ω and η adaptively
   ```

3. **Key Decision Points** (RL Opportunities):
   - Perturbation operator selection (Concentric vs Sequential)
   - Perturbation strength (omega value)
   - Acceptance criterion threshold (eta)
   - Local search intensity
   - Insertion heuristic selection

## Proposed RL Enhancement Strategies

### 1. **Deep Reinforcement Learning for Parameter Control**

#### A. Policy Network for Adaptive Parameter Selection
**Implementation**: Replace rule-based adaptive mechanisms with learned policies

**State Representation** (Input Features):
- Current solution quality (normalized gap from best known)
- Number of iterations since last improvement
- Current omega and eta values
- Distance metrics (distanceLS, obtainedDist)
- Solution diversity measures
- Time elapsed / iterations completed ratio
- Route statistics (number of routes, average route length, load distribution)
- Search trajectory features (improvement rate, stagnation indicator)

**Action Space**:
- Option 1: **Discrete Actions**
  - Omega adjustment: {increase_large, increase_small, keep, decrease_small, decrease_large}
  - Eta adjustment: {increase, keep, decrease}
  - Operator selection: {Concentric, Sequential}
  
- Option 2: **Continuous Actions**
  - Direct omega value in [omegaMin, omegaMax]
  - Direct eta value in feasible range
  
**Network Architecture**:
```
Input Layer (state features) 
   → Dense(256, ReLU) + Dropout(0.2)
   → Dense(128, ReLU) + Dropout(0.2)
   → Dense(64, ReLU)
   → Output Layers:
      - Actor: Dense(action_dim, Softmax/Tanh)
      - Critic: Dense(1, Linear) [for value estimation]
```

**Reward Function**:
```
R_t = α₁ * ΔQuality + α₂ * DiversityBonus + α₃ * EfficiencyPenalty

Where:
- ΔQuality = (f_old - f_new) / f_old  (normalized improvement)
- DiversityBonus = +bonus if exploring new regions, -penalty for cycling
- EfficiencyPenalty = -time_cost (encourage faster improvements)
```

#### B. Multi-Armed Bandit for Operator Selection
**Implementation**: UCB (Upper Confidence Bound) or Thompson Sampling

**Context**: Problem instance features
**Arms**: {Concentric, Sequential} × {InsertionHeuristic variants}

**Reward**: Immediate solution quality improvement after perturbation

**Advantages**:
- Simpler than full RL
- Fast adaptation within single instance
- Can be combined with deeper RL for cross-instance learning

### 2. **Graph Neural Network for Construction Heuristic**

#### A. Attention-Based Route Construction
**Inspiration**: Attention mechanism for sequence-to-sequence learning

**Architecture**:
```
Node Embeddings:
  - Concatenate [coordinates, demand, distance_to_depot, remaining_capacity]
  
Encoder:
  → Graph Attention Network (3 layers)
  → Node representations capture spatial and demand patterns
  
Decoder (autoregressive):
  → At each step, compute attention over unvisited nodes
  → Sample next node based on attention scores + mask (feasibility)
  → Update context with selected node
```

**Training**:
- Supervised learning on known good solutions
- REINFORCE with baseline (policy gradient)
- Beam search for inference

**Integration Point**: Replace or augment `ConstructSolution.construct()`

### 3. **Deep Q-Network (DQN) for Local Search Guidance**

#### A. Move Selection in Local Search
**Current**: Exhaustive search over neighborhood

**RL Enhancement**: Learn which moves are most promising

**State**: 
- Current route configuration
- Node features (position in route, slack capacity)
- Neighborhood move features

**Actions**: Type of move {Shift, Swap, 2-opt, Cross} × node pairs

**Q-Network**: Predicts expected improvement for each move type

**Advantages**:
- Reduces computational cost by pruning unpromising moves
- Learns problem-specific patterns across instances

### 4. **Curriculum Learning Strategy**

**Progressive Training Schedule**:
1. **Phase 1**: Small instances (n ≤ 50)
   - Train basic decision policies
   - Fast iterations, quick feedback
   
2. **Phase 2**: Medium instances (50 < n ≤ 100)
   - Refine policies with more complex patterns
   - Introduce harder constraints
   
3. **Phase 3**: Large instances (n > 100)
   - Fine-tune on realistic problem sizes
   - Transfer learning from smaller instances

### 5. **Hybrid Architecture: RL + AILS**

#### A. RL-AILS Framework
```java
public class RLAILSII extends AILSII {
    RLAgent policyAgent;
    StateExtractor stateExtractor;
    ReplayBuffer experienceBuffer;
    
    public void search() {
        // Initialize as before
        constructInitialSolution();
        
        while (!stoppingCriterion()) {
            // Extract state features
            State currentState = stateExtractor.extractState(
                solution, referenceSolution, iterator, timeElapsed
            );
            
            // RL policy decides actions
            Action action = policyAgent.selectAction(currentState);
            
            // Apply action (perturbation with RL-selected parameters)
            applyRLAction(action);
            
            // Standard AILS flow
            feasibilityPhase();
            localSearch();
            
            // Calculate reward
            double reward = computeReward(oldQuality, newQuality);
            
            // Store experience
            experienceBuffer.store(currentState, action, reward, newState);
            
            // Acceptance criterion (can also be RL-guided)
            if (acceptSolution()) {
                updateReferenceSolution();
            }
            
            // Periodic RL training
            if (iterator % UPDATE_FREQUENCY == 0) {
                policyAgent.train(experienceBuffer.sample());
            }
        }
    }
}
```

#### B. Two-Stage Training
**Stage 1 - Offline Learning**:
- Pre-train on diverse instances with known solutions
- Learn general patterns and strategies
- Save model checkpoints

**Stage 2 - Online Adaptation**:
- Fine-tune during actual problem solving
- Instance-specific adaptation
- Balance exploration vs exploitation

### 6. **Meta-Learning for Quick Adaptation**

**Goal**: Learn to quickly adapt to new problem instances

**MAML (Model-Agnostic Meta-Learning)** approach:
- Train on distribution of CVRP instances
- Learn initialization that adapts quickly with few gradient steps
- Fast fine-tuning on new instances

**Implementation**:
```python
# Meta-training loop
for meta_iteration in range(N_meta):
    # Sample batch of instances
    instances = sample_instances(batch_size)
    
    for instance in instances:
        # Inner loop: adapt to this instance
        theta_adapted = theta - α * ∇L(theta, instance)
        
        # Compute meta-loss on validation trajectories
        meta_loss += L(theta_adapted, instance_validation)
    
    # Outer loop: update meta-parameters
    theta = theta - β * ∇meta_loss
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
- [ ] Design state representation and feature extraction
- [ ] Implement basic RL environment wrapper for AILS-II
- [ ] Create experience replay buffer
- [ ] Develop reward function and evaluation metrics

### Phase 2: Simple RL Integration (Weeks 4-6)
- [ ] Implement Multi-Armed Bandit for operator selection
- [ ] Add contextual bandits with instance features
- [ ] Benchmark against baseline AILS-II
- [ ] Collect training data from multiple runs

### Phase 3: Deep RL Policy (Weeks 7-10)
- [ ] Implement Actor-Critic policy network
- [ ] Train on small instances (curriculum learning)
- [ ] Integrate with main search loop
- [ ] Evaluate on benchmark instances

### Phase 4: Advanced Features (Weeks 11-14)
- [ ] Add GNN for construction heuristic
- [ ] Implement DQN for local search guidance
- [ ] Meta-learning for fast adaptation
- [ ] Hyperparameter tuning

### Phase 5: Evaluation & Refinement (Weeks 15-16)
- [ ] Comprehensive benchmarking on all datasets
- [ ] Statistical significance testing
- [ ] Ablation studies (which components help most)
- [ ] Documentation and paper writing

## Technical Requirements

### Software Stack
**Python Components** (RL Training):
```
- PyTorch 2.0+ (deep learning framework)
- PyTorch Geometric (for GNN)
- Stable-Baselines3 (RL algorithms)
- Ray RLlib (distributed training)
- NumPy, Pandas (data handling)
- Weights & Biases (experiment tracking)
```

**Java-Python Bridge**:
```
- Py4J or JPype (Python-Java interop)
- Protocol Buffers (efficient serialization)
- RESTful API (if decoupled architecture)
```

### Hardware Recommendations
- **Training**: GPU (NVIDIA RTX 3090 or better) for neural network training
- **Inference**: CPU sufficient for deployment
- **Memory**: 16GB+ RAM for large instances

## Performance Metrics

### Evaluation Criteria
1. **Solution Quality**: Gap from best known solutions (BKS)
2. **Convergence Speed**: Time/iterations to reach target quality
3. **Robustness**: Performance across different instance types
4. **Generalization**: Transfer to unseen instances
5. **Computational Efficiency**: Runtime overhead of RL components

### Baseline Comparisons
- AILS-II (current implementation)
- LKH3 (state-of-art heuristic)
- HGS-CVRP (hybrid genetic search)
- Other RL-based methods (if available)

## Key Research Questions

1. **Which decisions benefit most from RL?**
   - High-level strategy (parameter control) vs low-level tactics (move selection)?

2. **Sample efficiency**: 
   - How much training data needed for good performance?
   - Can we learn from failed attempts?

3. **Generalization**:
   - Does RL learned on small instances transfer to large ones?
   - Cross-benchmark generalization?

4. **Interpretability**:
   - Can we understand what the RL agent learns?
   - Extract rules from learned policies?

## Potential Challenges & Solutions

### Challenge 1: Long Training Time
**Solution**: 
- Start with simpler RL methods (bandits, simple policies)
- Use distributed training (Ray, multiple GPUs)
- Warm-start with supervised learning on expert demonstrations

### Challenge 2: Reward Engineering
**Solution**:
- Multi-objective reward with careful weighting
- Reward shaping based on domain knowledge
- Inverse RL to learn from expert (AILS-II) trajectories

### Challenge 3: Java-Python Integration
**Solution**:
- Option A: Rewrite RL components in Java (use DL4J, DeepNetts)
- Option B: Efficient Python-Java bridge with minimal overhead
- Option C: Separate RL training (Python) and deployment (Java integration)

### Challenge 4: Overfitting to Training Instances
**Solution**:
- Large, diverse training set
- Data augmentation (rotation, scaling)
- Regularization (dropout, weight decay)
- Cross-validation on held-out instances

## Expected Outcomes

### Optimistic Scenario (Success)
- 2-5% improvement in solution quality on benchmark instances
- 20-30% faster convergence to good solutions
- Strong generalization to unseen instances
- Publishable results in top-tier journals (EJOR, COR, IJOC)

### Realistic Scenario (Partial Success)
- 1-2% improvement on some instance classes
- RL provides complementary strengths to AILS-II
- Better understanding of which decisions benefit from learning
- Solid conference paper or workshop publication

### Learning Outcomes (Even if Numbers Don't Improve)
- Deep insights into CVRP structure
- Reusable RL framework for other combinatorial problems
- Identified failure modes and future research directions
- Strong foundation for next-generation methods

## References & Related Work

### Key Papers
1. **Attention-based models**:
   - Kool et al. "Attention, Learn to Solve Routing Problems!" (2019)
   - Vinyals et al. "Pointer Networks" (2015)

2. **RL for Combinatorial Optimization**:
   - Bello et al. "Neural Combinatorial Optimization with RL" (2017)
   - Nazari et al. "Reinforcement Learning for Solving VRP" (2018)

3. **Hybrid Approaches**:
   - Lu et al. "Learning to Delegate for Large-scale Vehicle Routing" (2020)
   - Hottung & Tierney "Neural Large Neighborhood Search" (2020)

4. **Graph Neural Networks**:
   - Joshi et al. "Learning TSP Requires Rethinking Generalization" (2021)
   - Ma et al. "Learning to Iteratively Solve Routing Problems" (2020)

### Code Resources
- [rl4co](https://github.com/ai4co/rl4co): RL library for CO problems
- [attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route): Reference implementation
- [OR-Tools](https://developers.google.com/optimization): Google's optimization library

## Next Steps

1. **Literature Review**: Deep dive into recent RL+CVRP papers
2. **Prototype**: Simple bandit-based operator selection (quick win)
3. **Data Collection**: Run AILS-II with extensive logging for analysis
4. **Design Document**: Detailed architecture for RL-AILS integration
5. **Proof of Concept**: Minimal RL agent on toy instances

---

**Document Version**: 1.0  
**Date**: February 5, 2026  
**Branch**: rl-improvement  
**Status**: Planning Phase
