package ML;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import Data.Instance;
import Data.Node;

import java.nio.file.Paths;

/**
 * GNN-based Construction Heuristic using Deep Java Library (DJL)
 * 
 * Loads a pre-trained PyTorch GNN model and uses it to generate
 * initial CVRP solutions with deep learning.
 * 
 * Performance: ~1-5ms per inference (Java native speed)
 */
public class GNNConstructionHeuristic {
    
    private Model model;
    private Predictor<CVRPInput, int[]> predictor;
    private boolean initialized = false;
    
    /**
     * Initialize the GNN model from file
     * 
     * @param modelPath Path to the TorchScript model (.pt file)
     */
    public GNNConstructionHeuristic(String modelPath) {
        try {
            // Create model
            model = Model.newInstance("gnn_cvrp");
            model.load(Paths.get(modelPath));
            
            // Create predictor with custom translator
            CVRPTranslator translator = new CVRPTranslator();
            predictor = model.newPredictor(translator);
            
            initialized = true;
            System.out.println("GNN model loaded successfully from: " + modelPath);
            
        } catch (Exception e) {
            System.err.println("Failed to load GNN model: " + e.getMessage());
            System.err.println("Falling back to classical construction heuristic");
            initialized = false;
        }
    }
    
    /**
     * Generate initial solution using GNN
     * 
     * @param instance CVRP instance
     * @return Array of node indices representing visit order
     */
    public int[] constructSolution(Instance instance) {
        if (!initialized) {
            return null; // Fall back to classical heuristic
        }
        
        try {
            // Prepare input
            CVRPInput input = new CVRPInput(instance);
            
            // Inference
            long startTime = System.nanoTime();
            int[] tour = predictor.predict(input);
            long endTime = System.nanoTime();
            
            double inferenceTimeMs = (endTime - startTime) / 1_000_000.0;
            System.out.println("GNN inference time: " + String.format("%.2f", inferenceTimeMs) + " ms");
            
            return tour;
            
        } catch (Exception e) {
            System.err.println("GNN inference failed: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Check if GNN is available and initialized
     */
    public boolean isAvailable() {
        return initialized;
    }
    
    /**
     * Close resources
     */
    public void close() {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }
    
    /**
     * Input data structure for CVRP instance
     */
    static class CVRPInput {
        float[][] nodeFeatures;  // [num_nodes, 4] - x, y, demand, dist_to_depot
        int numNodes;
        double capacity;
        
        CVRPInput(Instance instance) {
            this.numNodes = instance.getSize();
            this.capacity = instance.getCapacity();
            this.nodeFeatures = new float[numNodes][4];
            
            // Normalize coordinates to [0, 1]
            double maxX = Double.MIN_VALUE, maxY = Double.MIN_VALUE;
            double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE;
            
            for (int i = 0; i < numNodes; i++) {
                Node node = instance.getNode(i);
                maxX = Math.max(maxX, node.x);
                maxY = Math.max(maxY, node.y);
                minX = Math.min(minX, node.x);
                minY = Math.min(minY, node.y);
            }
            
            double rangeX = maxX - minX;
            double rangeY = maxY - minY;
            
            Node depot = instance.getNode(0);
            
            for (int i = 0; i < numNodes; i++) {
                Node node = instance.getNode(i);
                
                // Normalized x, y coordinates
                nodeFeatures[i][0] = (float) ((node.x - minX) / rangeX);
                nodeFeatures[i][1] = (float) ((node.y - minY) / rangeY);
                
                // Normalized demand
                nodeFeatures[i][2] = (float) (node.demand / capacity);
                
                // Normalized distance to depot
                double dist = Math.sqrt(
                    Math.pow(node.x - depot.x, 2) + 
                    Math.pow(node.y - depot.y, 2)
                );
                double maxDist = Math.sqrt(rangeX * rangeX + rangeY * rangeY);
                nodeFeatures[i][3] = (float) (dist / maxDist);
            }
        }
    }
    
    /**
     * Translator between Java objects and NDArrays
     */
    static class CVRPTranslator implements Translator<CVRPInput, int[]> {
        
        @Override
        public NDList processInput(TranslatorContext ctx, CVRPInput input) {
            NDManager manager = ctx.getNDManager();
            
            // Convert features to NDArray: [1, num_nodes, 4]
            // Batch size = 1 (single instance inference)
            NDArray ndFeatures = manager.create(input.nodeFeatures);
            ndFeatures = ndFeatures.expandDims(0);  // Add batch dimension
            
            return new NDList(ndFeatures);
        }
        
        @Override
        public int[] processOutput(TranslatorContext ctx, NDList list) {
            // Output: [1, num_nodes] tour sequence
            NDArray output = list.singletonOrThrow();
            
            // Remove batch dimension and convert to int array
            output = output.squeeze(0);
            long[] longTour = output.toLongArray();
            
            int[] tour = new int[longTour.length];
            for (int i = 0; i < longTour.length; i++) {
                tour[i] = (int) longTour[i];
            }
            
            return tour;
        }
        
        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }
}
