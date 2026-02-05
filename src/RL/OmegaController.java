package RL;

import java.util.Random;

/**
 * RL-based controller for omega (perturbation strength) adjustment.
 * Uses Q-learning with discretized state-action space to learn when to
 * increase, decrease, or maintain omega based on search progress.
 */
public class OmegaController {
    
    // State discretization
    private static final int NUM_IMPROVEMENT_STATES = 3; // stagnant, slow, good
    private static final int NUM_OMEGA_STATES = 3; // low, medium, high
    private static final int NUM_STATES = NUM_IMPROVEMENT_STATES * NUM_OMEGA_STATES;
    
    // Actions: decrease omega, keep omega, increase omega
    private static final int ACTION_DECREASE = 0;
    private static final int ACTION_KEEP = 1;
    private static final int ACTION_INCREASE = 2;
    private static final int NUM_ACTIONS = 3;
    
    // Q-learning parameters
    private double[][] qTable;
    private double learningRate; // alpha
    private double discountFactor; // gamma
    private double epsilon; // exploration rate
    private Random rand;
    
    // Omega bounds
    private double omegaMin;
    private double omegaMax;
    private double omegaStep; // step size for increase/decrease
    
    // State tracking
    private int iterationsSinceImprovement;
    private double recentImprovementRate; // improvement per iteration
    private int lastState;
    private int lastAction;
    private boolean enabled;
    
    // For logging
    private int[] actionCounts;
    private double totalReward;
    
    /**
     * Constructor for OmegaController
     * 
     * @param omegaMin Minimum omega value
     * @param omegaMax Maximum omega value
     * @param learningRate Q-learning alpha (default: 0.1)
     * @param discountFactor Q-learning gamma (default: 0.9)
     * @param epsilon Exploration rate (default: 0.1)
     * @param seed Random seed (null for non-deterministic)
     * @param enabled Whether RL control is enabled
     */
    public OmegaController(double omegaMin, double omegaMax, double learningRate,
                          double discountFactor, double epsilon, Long seed, boolean enabled) {
        this.omegaMin = omegaMin;
        this.omegaMax = omegaMax;
        this.omegaStep = (omegaMax - omegaMin) / 10.0; // 10% steps
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.epsilon = epsilon;
        this.enabled = enabled;
        
        // Initialize Q-table with optimistic values to encourage exploration
        this.qTable = new double[NUM_STATES][NUM_ACTIONS];
        for (int s = 0; s < NUM_STATES; s++) {
            for (int a = 0; a < NUM_ACTIONS; a++) {
                qTable[s][a] = 0.0; // Start neutral
            }
        }
        
        // Initialize random
        if (seed != null) {
            this.rand = new Random(seed);
        } else {
            this.rand = new Random();
        }
        
        // Initialize tracking
        this.iterationsSinceImprovement = 0;
        this.recentImprovementRate = 0.0;
        this.lastState = -1;
        this.lastAction = -1;
        this.actionCounts = new int[NUM_ACTIONS];
        this.totalReward = 0.0;
    }
    
    /**
     * Discretize the current search state into a state index
     * 
     * @param iterationsSinceImprovement Iterations without improvement
     * @param currentOmega Current omega value
     * @return State index (0 to NUM_STATES-1)
     */
    private int discretizeState(int iterationsSinceImprovement, double currentOmega) {
        // Improvement state: 0=good (<10 iters), 1=slow (10-50), 2=stagnant (>50)
        int improvementState;
        if (iterationsSinceImprovement < 10) {
            improvementState = 0; // good
        } else if (iterationsSinceImprovement < 50) {
            improvementState = 1; // slow
        } else {
            improvementState = 2; // stagnant
        }
        
        // Omega state: 0=low, 1=medium, 2=high
        int omegaState;
        double omegaRange = omegaMax - omegaMin;
        double normalizedOmega = (currentOmega - omegaMin) / omegaRange;
        if (normalizedOmega < 0.33) {
            omegaState = 0; // low
        } else if (normalizedOmega < 0.67) {
            omegaState = 1; // medium
        } else {
            omegaState = 2; // high
        }
        
        return improvementState * NUM_OMEGA_STATES + omegaState;
    }
    
    /**
     * Select action (omega adjustment) using epsilon-greedy policy
     * 
     * @param iterationsSinceImprovement Iterations without improvement
     * @param currentOmega Current omega value
     * @return Action index (0=decrease, 1=keep, 2=increase)
     */
    public int selectAction(int iterationsSinceImprovement, double currentOmega) {
        if (!enabled) {
            return ACTION_KEEP; // No RL, keep omega as-is
        }
        
        int state = discretizeState(iterationsSinceImprovement, currentOmega);
        this.lastState = state;
        
        int action;
        
        // Epsilon-greedy: explore with probability epsilon
        if (rand.nextDouble() < epsilon) {
            // Random action (exploration)
            action = rand.nextInt(NUM_ACTIONS);
        } else {
            // Greedy action (exploitation)
            action = getBestAction(state);
        }
        
        this.lastAction = action;
        this.iterationsSinceImprovement = iterationsSinceImprovement;
        this.actionCounts[action]++;
        
        return action;
    }
    
    /**
     * Get the best action for a given state (highest Q-value)
     */
    private int getBestAction(int state) {
        int bestAction = 0;
        double bestQ = qTable[state][0];
        
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qTable[state][a] > bestQ) {
                bestQ = qTable[state][a];
                bestAction = a;
            }
        }
        
        return bestAction;
    }
    
    /**
     * Apply the selected action to omega
     * 
     * @param currentOmega Current omega value
     * @param action Action to apply
     * @return New omega value
     */
    public double applyAction(double currentOmega, int action) {
        if (!enabled || action == ACTION_KEEP) {
            return currentOmega;
        }
        
        double newOmega = currentOmega;
        
        if (action == ACTION_DECREASE) {
            newOmega = Math.max(omegaMin, currentOmega - omegaStep);
        } else if (action == ACTION_INCREASE) {
            newOmega = Math.min(omegaMax, currentOmega + omegaStep);
        }
        
        return newOmega;
    }
    
    /**
     * Update Q-table with reward feedback (Q-learning update)
     * 
     * @param reward Immediate reward
     * @param newIterationsSinceImprovement New state's iterations without improvement
     * @param newOmega New omega value
     */
    public void updateQ(double reward, int newIterationsSinceImprovement, double newOmega) {
        if (!enabled || lastState == -1 || lastAction == -1) {
            return;
        }
        
        int newState = discretizeState(newIterationsSinceImprovement, newOmega);
        
        // Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        double currentQ = qTable[lastState][lastAction];
        double maxNextQ = getMaxQ(newState);
        double tdError = reward + discountFactor * maxNextQ - currentQ;
        qTable[lastState][lastAction] = currentQ + learningRate * tdError;
        
        this.totalReward += reward;
    }
    
    /**
     * Get maximum Q-value for a state
     */
    private double getMaxQ(int state) {
        double maxQ = qTable[state][0];
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (qTable[state][a] > maxQ) {
                maxQ = qTable[state][a];
            }
        }
        return maxQ;
    }
    
    /**
     * Calculate reward based on solution improvement
     * Reward should encourage actions that lead to improvements
     * 
     * @param previousQuality Quality before omega adjustment
     * @param newQuality Quality after perturbation and local search
     * @param foundNewBest Whether this iteration found a new global best
     * @return Reward value
     */
    public static double calculateReward(double previousQuality, double newQuality, boolean foundNewBest) {
        double baseReward = (previousQuality - newQuality) / previousQuality;
        
        // Large bonus for finding new best solution
        if (foundNewBest) {
            baseReward += 0.1; // 10% bonus
        }
        
        return baseReward;
    }
    
    /**
     * Get statistics string
     */
    public String getStats() {
        if (!enabled) {
            return "Omega RL Control: DISABLED";
        }
        
        StringBuilder sb = new StringBuilder();
        sb.append("\n=== Omega RL Controller Statistics ===\n");
        sb.append(String.format("Actions: Decrease=%d, Keep=%d, Increase=%d\n",
                                actionCounts[ACTION_DECREASE],
                                actionCounts[ACTION_KEEP],
                                actionCounts[ACTION_INCREASE]));
        sb.append(String.format("Total Reward: %.6f\n", totalReward));
        sb.append(String.format("Avg Reward: %.6f\n", totalReward / (actionCounts[0] + actionCounts[1] + actionCounts[2])));
        
        // Show Q-table summary
        sb.append("\nQ-Table Summary (State → Best Action):\n");
        String[] improvementLabels = {"Good", "Slow", "Stagnant"};
        String[] omegaLabels = {"Low", "Med", "High"};
        String[] actionLabels = {"Decrease", "Keep", "Increase"};
        
        for (int imp = 0; imp < NUM_IMPROVEMENT_STATES; imp++) {
            for (int omg = 0; omg < NUM_OMEGA_STATES; omg++) {
                int state = imp * NUM_OMEGA_STATES + omg;
                int bestAction = getBestAction(state);
                double bestQ = qTable[state][bestAction];
                sb.append(String.format("  [%s, Omega=%s]: %s (Q=%.4f)\n",
                                      improvementLabels[imp],
                                      omegaLabels[omg],
                                      actionLabels[bestAction],
                                      bestQ));
            }
        }
        
        return sb.toString();
    }
    
    /**
     * Get action name for logging
     */
    public static String getActionName(int action) {
        switch (action) {
            case ACTION_DECREASE: return "DECREASE";
            case ACTION_KEEP: return "KEEP";
            case ACTION_INCREASE: return "INCREASE";
            default: return "UNKNOWN";
        }
    }
    
    /**
     * Check if RL control is enabled
     */
    public boolean isEnabled() {
        return enabled;
    }
    
    /**
     * Get last selected action
     */
    public int getLastAction() {
        return lastAction;
    }
}
