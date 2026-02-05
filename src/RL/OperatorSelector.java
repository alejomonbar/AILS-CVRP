package RL;

import java.util.Random;
import Perturbation.Perturbation;

/**
 * Multi-Armed Bandit (UCB1) based operator selector for perturbation operators.
 * Uses Upper Confidence Bound algorithm to balance exploration and exploitation.
 */
public class OperatorSelector {
    
    private Perturbation[] operators;
    private int[] selectionCounts;  // Number of times each operator was selected
    private double[] cumulativeRewards;  // Sum of rewards for each operator
    private int totalSelections;
    private double explorationParameter;  // UCB exploration parameter (default: sqrt(2))
    private Random rand;
    private boolean enabled;
    
    // For tracking
    private int lastSelectedIndex;
    private double lastReward;
    
    /**
     * Constructor for OperatorSelector
     * @param operators Array of perturbation operators to choose from
     * @param explorationParameter UCB exploration parameter (typically sqrt(2) ≈ 1.414)
     * @param seed Random seed for reproducibility (null for non-deterministic)
     * @param enabled Whether RL-based selection is enabled (false = random selection)
     */
    public OperatorSelector(Perturbation[] operators, double explorationParameter, Long seed, boolean enabled) {
        this.operators = operators;
        this.explorationParameter = explorationParameter;
        this.enabled = enabled;
        this.totalSelections = 0;
        this.lastSelectedIndex = -1;
        this.lastReward = 0.0;
        
        // Initialize tracking arrays
        this.selectionCounts = new int[operators.length];
        this.cumulativeRewards = new double[operators.length];
        
        // Initialize random
        if (seed != null) {
            this.rand = new Random(seed);
        } else {
            this.rand = new Random();
        }
        
        // Initialize with one selection per operator (avoid division by zero)
        for (int i = 0; i < operators.length; i++) {
            selectionCounts[i] = 1;
            cumulativeRewards[i] = 0.0;
        }
        this.totalSelections = operators.length;
    }
    
    /**
     * Select an operator using UCB1 algorithm or random selection
     * @return Selected perturbation operator
     */
    public Perturbation selectOperator() {
        if (!enabled) {
            // Random selection (baseline)
            lastSelectedIndex = rand.nextInt(operators.length);
            return operators[lastSelectedIndex];
        }
        
        // UCB1 algorithm: select operator with highest upper confidence bound
        double bestUCB = Double.NEGATIVE_INFINITY;
        int bestIndex = 0;
        
        for (int i = 0; i < operators.length; i++) {
            double avgReward = cumulativeRewards[i] / selectionCounts[i];
            double exploration = explorationParameter * Math.sqrt(Math.log(totalSelections) / selectionCounts[i]);
            double ucb = avgReward + exploration;
            
            if (ucb > bestUCB) {
                bestUCB = ucb;
                bestIndex = i;
            }
        }
        
        lastSelectedIndex = bestIndex;
        selectionCounts[bestIndex]++;
        totalSelections++;
        
        return operators[bestIndex];
    }
    
    /**
     * Provide reward feedback for the last selected operator
     * @param reward Reward value (higher is better)
     */
    public void giveReward(double reward) {
        if (lastSelectedIndex >= 0 && enabled) {
            cumulativeRewards[lastSelectedIndex] += reward;
            lastReward = reward;
        }
    }
    
    /**
     * Calculate reward based on solution improvement
     * Reward = (previousQuality - newQuality) / previousQuality
     * Positive reward for improvement, negative for deterioration
     * 
     * @param previousQuality Quality before perturbation
     * @param newQuality Quality after perturbation and local search
     * @return Normalized reward value
     */
    public static double calculateImprovementReward(double previousQuality, double newQuality) {
        if (previousQuality == 0) return 0.0;
        return (previousQuality - newQuality) / previousQuality;
    }
    
    /**
     * Get statistics for a specific operator
     * @param index Operator index
     * @return String with selection count, avg reward, and UCB value
     */
    public String getOperatorStats(int index) {
        if (index < 0 || index >= operators.length) return "Invalid index";
        
        double avgReward = cumulativeRewards[index] / selectionCounts[index];
        double exploration = explorationParameter * Math.sqrt(Math.log(totalSelections) / selectionCounts[index]);
        double ucb = avgReward + exploration;
        
        return String.format("Operator %d: selections=%d, avgReward=%.6f, UCB=%.6f", 
                           index, selectionCounts[index], avgReward, ucb);
    }
    
    /**
     * Get the index of the last selected operator
     */
    public int getLastSelectedIndex() {
        return lastSelectedIndex;
    }
    
    /**
     * Get the last reward given
     */
    public double getLastReward() {
        return lastReward;
    }
    
    /**
     * Get the name of an operator by index
     */
    public String getOperatorName(int index) {
        if (index < 0 || index >= operators.length) return "Unknown";
        return operators[index].getClass().getSimpleName();
    }
    
    /**
     * Get all statistics as a formatted string
     */
    public String getAllStats() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n=== Operator Selection Statistics ===\n");
        sb.append(String.format("Total selections: %d\n", totalSelections));
        sb.append(String.format("RL enabled: %b\n", enabled));
        for (int i = 0; i < operators.length; i++) {
            sb.append(getOperatorStats(i)).append("\n");
        }
        return sb.toString();
    }
    
    /**
     * Get selection counts for logging
     */
    public int[] getSelectionCounts() {
        return selectionCounts.clone();
    }
    
    /**
     * Get average rewards for logging
     */
    public double[] getAverageRewards() {
        double[] avgRewards = new double[operators.length];
        for (int i = 0; i < operators.length; i++) {
            avgRewards[i] = cumulativeRewards[i] / selectionCounts[i];
        }
        return avgRewards;
    }
    
    /**
     * Check if RL is enabled
     */
    public boolean isEnabled() {
        return enabled;
    }
}
