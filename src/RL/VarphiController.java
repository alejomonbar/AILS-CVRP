package RL;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Q-Learning based controller for adaptive varphi (neighborhood size) adjustment.
 * 
 * State Space:
 * - Search Progress: Early (0-33%), Mid (33-67%), Late (67-100%)
 * - Stagnation Level: Active (recent improvement), Slow (some stagnation), Stuck (high stagnation)
 * 
 * Action Space:
 * - DECREASE: Reduce varphi for faster, focused search
 * - KEEP: Maintain current varphi
 * - INCREASE: Increase varphi for more thorough search
 * 
 * Reward:
 * - Improvement ratio (quality change / previous quality)
 * - Time penalty (discourage large varphi if not improving)
 * - Stagnation break bonus (reward for escaping local optima)
 */
public class VarphiController {
    
    public enum Action {
        DECREASE, KEEP, INCREASE
    }
    
    private enum SearchPhase {
        EARLY, MID, LATE
    }
    
    private enum StagnationLevel {
        ACTIVE,  // Recent improvements
        SLOW,    // Some stagnation
        STUCK    // High stagnation
    }
    
    // Q-learning parameters
    private double alpha;      // Learning rate
    private double gamma;      // Discount factor
    private double epsilon;    // Exploration rate
    
    // State-action Q-table
    private Map<String, Map<Action, Double>> qTable;
    
    // Varphi bounds
    private final int varphiMin;
    private final int varphiMax;
    private final int varphiDefault;
    private int currentVarphi;
    
    // State tracking
    private int totalIterations;
    private int iterationsSinceImprovement;
    private double previousQuality;
    private boolean isFirstUpdate;
    
    // Statistics
    private Map<Action, Integer> actionCounts;
    private double totalReward;
    private int updateCount;
    
    // Time tracking for penalty calculation
    private long lastIterationTime;
    
    private Random random;
    
    public VarphiController(int varphiDefault, double alpha, double gamma, double epsilon, Random random) {
        this.varphiDefault = varphiDefault;
        this.varphiMin = Math.max(10, varphiDefault / 4);  // At least 10, or 1/4 of default
        this.varphiMax = Math.min(100, varphiDefault * 2); // At most 100, or 2x default
        this.currentVarphi = varphiDefault;
        
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.random = random;
        
        this.qTable = new HashMap<>();
        initializeQTable();
        
        this.actionCounts = new HashMap<>();
        for (Action action : Action.values()) {
            actionCounts.put(action, 0);
        }
        
        this.totalReward = 0.0;
        this.updateCount = 0;
        this.totalIterations = 0;
        this.iterationsSinceImprovement = 0;
        this.isFirstUpdate = true;
        this.lastIterationTime = System.currentTimeMillis();
    }
    
    private void initializeQTable() {
        for (SearchPhase phase : SearchPhase.values()) {
            for (StagnationLevel stag : StagnationLevel.values()) {
                String state = getStateKey(phase, stag);
                Map<Action, Double> actions = new HashMap<>();
                for (Action action : Action.values()) {
                    actions.put(action, 0.0);
                }
                qTable.put(state, actions);
            }
        }
    }
    
    private String getStateKey(SearchPhase phase, StagnationLevel stag) {
        return phase.name() + "_" + stag.name();
    }
    
    private SearchPhase getSearchPhase(int currentIter, int maxIter) {
        double progress = (double) currentIter / maxIter;
        if (progress < 0.33) return SearchPhase.EARLY;
        if (progress < 0.67) return SearchPhase.MID;
        return SearchPhase.LATE;
    }
    
    private StagnationLevel getStagnationLevel() {
        if (iterationsSinceImprovement < 50) return StagnationLevel.ACTIVE;
        if (iterationsSinceImprovement < 150) return StagnationLevel.SLOW;
        return StagnationLevel.STUCK;
    }
    
    /**
     * Select action using epsilon-greedy policy
     */
    public Action selectAction(int currentIteration, int maxIterations) {
        String state = getCurrentState(currentIteration, maxIterations);
        
        // Epsilon-greedy exploration
        if (random.nextDouble() < epsilon) {
            // Explore: random action
            Action[] actions = Action.values();
            return actions[random.nextInt(actions.length)];
        } else {
            // Exploit: best action
            return getBestAction(state);
        }
    }
    
    private String getCurrentState(int currentIteration, int maxIterations) {
        SearchPhase phase = getSearchPhase(currentIteration, maxIterations);
        StagnationLevel stag = getStagnationLevel();
        return getStateKey(phase, stag);
    }
    
    private Action getBestAction(String state) {
        Map<Action, Double> actions = qTable.get(state);
        Action bestAction = Action.KEEP;
        double bestValue = Double.NEGATIVE_INFINITY;
        
        for (Map.Entry<Action, Double> entry : actions.entrySet()) {
            if (entry.getValue() > bestValue) {
                bestValue = entry.getValue();
                bestAction = entry.getKey();
            }
        }
        
        return bestAction;
    }
    
    /**
     * Apply the selected action to varphi
     */
    public int applyAction(Action action) {
        actionCounts.put(action, actionCounts.get(action) + 1);
        
        int step = (varphiMax - varphiMin) / 10; // 10% adjustment
        step = Math.max(1, step); // At least 1
        
        switch (action) {
            case DECREASE:
                currentVarphi = Math.max(varphiMin, currentVarphi - step);
                break;
            case INCREASE:
                currentVarphi = Math.min(varphiMax, currentVarphi + step);
                break;
            case KEEP:
                // No change
                break;
        }
        
        return currentVarphi;
    }
    
    /**
     * Update Q-values based on observed reward
     */
    public void updateQ(int previousIteration, int currentIteration, int maxIterations, 
                       Action action, double currentQuality, boolean foundNewBest) {
        
        if (isFirstUpdate) {
            previousQuality = currentQuality;
            isFirstUpdate = false;
            return;
        }
        
        // Calculate reward
        double reward = calculateReward(currentQuality, foundNewBest);
        totalReward += reward;
        updateCount++;
        
        // Get states
        String prevState = getCurrentState(previousIteration, maxIterations);
        String currState = getCurrentState(currentIteration, maxIterations);
        
        // Get current Q-value
        double currentQ = qTable.get(prevState).get(action);
        
        // Get max Q-value for next state
        double maxNextQ = getMaxQValue(currState);
        
        // Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        double newQ = currentQ + alpha * (reward + gamma * maxNextQ - currentQ);
        qTable.get(prevState).put(action, newQ);
        
        // Update state tracking
        previousQuality = currentQuality;
        totalIterations++;
        
        // Update stagnation counter
        if (currentQuality < previousQuality - 0.001) { // Small epsilon for floating point
            iterationsSinceImprovement = 0;
        } else {
            iterationsSinceImprovement++;
        }
        
        lastIterationTime = System.currentTimeMillis();
    }
    
    private double calculateReward(double currentQuality, boolean foundNewBest) {
        // Improvement ratio (negative because we minimize)
        double improvementRatio = (previousQuality - currentQuality) / previousQuality;
        
        // Base reward: improvement ratio
        double reward = improvementRatio * 100.0; // Scale up for better learning
        
        // Bonus for finding new best solution
        if (foundNewBest) {
            reward += 0.5; // Significant bonus
        }
        
        // Bonus for breaking out of stagnation
        if (improvementRatio > 0.001 && iterationsSinceImprovement > 100) {
            reward += 0.3; // Reward for escaping stagnation
        }
        
        // Small penalty for using large varphi (time cost)
        double varphiUsage = (double) (currentVarphi - varphiMin) / (varphiMax - varphiMin);
        reward -= varphiUsage * 0.05; // Small time penalty
        
        return reward;
    }
    
    private double getMaxQValue(String state) {
        Map<Action, Double> actions = qTable.get(state);
        double maxQ = Double.NEGATIVE_INFINITY;
        for (Double qValue : actions.values()) {
            if (qValue > maxQ) {
                maxQ = qValue;
            }
        }
        return maxQ;
    }
    
    public int getCurrentVarphi() {
        return currentVarphi;
    }
    
    public void printStatistics() {
        System.out.println("\n=== Varphi RL Controller Statistics ===");
        System.out.println("Actions: Decrease=" + actionCounts.get(Action.DECREASE) 
                         + ", Keep=" + actionCounts.get(Action.KEEP)
                         + ", Increase=" + actionCounts.get(Action.INCREASE));
        System.out.println("Total Reward: " + String.format("%.6f", totalReward));
        System.out.println("Avg Reward: " + String.format("%.6f", updateCount > 0 ? totalReward / updateCount : 0));
        System.out.println("Varphi Range: [" + varphiMin + ", " + varphiMax + "], Default: " + varphiDefault);
        System.out.println("Final Varphi: " + currentVarphi);
        
        System.out.println("\nQ-Table Summary (State → Best Action):");
        for (SearchPhase phase : SearchPhase.values()) {
            for (StagnationLevel stag : StagnationLevel.values()) {
                String state = getStateKey(phase, stag);
                Action bestAction = getBestAction(state);
                double bestQ = qTable.get(state).get(bestAction);
                System.out.println(String.format("  [%s, %s]: %s (Q=%.4f)", 
                    phase.name(), stag.name(), bestAction.name(), bestQ));
            }
        }
    }
    
    public Map<Action, Integer> getActionCounts() {
        return new HashMap<>(actionCounts);
    }
    
    public double getTotalReward() {
        return totalReward;
    }
    
    public int getUpdateCount() {
        return updateCount;
    }
}
