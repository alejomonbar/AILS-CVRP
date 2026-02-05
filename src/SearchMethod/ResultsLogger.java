package SearchMethod;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Tracks and saves experimental results for reproducibility and analysis
 */
public class ResultsLogger {
    private String outputDir;
    private String instanceName;
    private String runId;
    private BufferedWriter iterationWriter;
    private BufferedWriter summaryWriter;
    private BufferedWriter configWriter;
    private long startTime;
    
    public ResultsLogger(String outputDir, String instanceFile, Config config) {
        this.outputDir = outputDir;
        this.instanceName = extractInstanceName(instanceFile);
        this.runId = generateRunId();
        this.startTime = System.currentTimeMillis();
        
        // Create output directory if doesn't exist
        new File(outputDir).mkdirs();
        
        // Create run-specific subdirectory
        String runDir = outputDir + "/" + instanceName + "/" + runId;
        new File(runDir).mkdirs();
        
        try {
            // Create writers for different output files
            iterationWriter = new BufferedWriter(new FileWriter(runDir + "/iterations.csv"));
            summaryWriter = new BufferedWriter(new FileWriter(runDir + "/summary.csv"));
            configWriter = new BufferedWriter(new FileWriter(runDir + "/config.txt"));
            
            // Write headers
            writeIterationHeader();
            writeSummaryHeader();
            writeConfig(config, instanceFile);
            
        } catch (IOException e) {
            System.err.println("Error creating results logger: " + e.getMessage());
        }
    }
    
    private String extractInstanceName(String filepath) {
        String name = new File(filepath).getName();
        if (name.endsWith(".vrp")) {
            return name.substring(0, name.length() - 4);
        }
        return name;
    }
    
    private String generateRunId() {
        SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
        return sdf.format(new Date());
    }
    
    private void writeIterationHeader() throws IOException {
        iterationWriter.write("iteration,time_sec,solution_quality,gap_percent,num_routes,eta,omega,");
        iterationWriter.write("perturbation_type,insertion_heuristic,distance_ls,improvement\n");
        iterationWriter.flush();
    }
    
    private void writeSummaryHeader() throws IOException {
        summaryWriter.write("run_id,instance,seed,stopping_criterion,time_limit_or_iterations,");
        summaryWriter.write("final_quality,final_gap,best_quality,best_gap,");
        summaryWriter.write("total_iterations,time_to_best,iter_to_best,total_time\n");
        summaryWriter.flush();
    }
    
    private void writeConfig(Config config, String instanceFile) throws IOException {
        configWriter.write("=== Experiment Configuration ===\n");
        configWriter.write("Run ID: " + runId + "\n");
        configWriter.write("Instance: " + instanceFile + "\n");
        configWriter.write("Timestamp: " + new Date() + "\n\n");
        configWriter.write(config.toString() + "\n");
        configWriter.flush();
    }
    
    public void logIteration(int iteration, double timeSec, double quality, double gap, 
                             int numRoutes, double eta, double omega,
                             String perturbationType, String insertionHeuristic,
                             double distanceLS, double improvement) {
        try {
            iterationWriter.write(String.format("%d,%.3f,%.2f,%.4f,%d,%.4f,%.4f,%s,%s,%.4f,%.4f\n",
                iteration, timeSec, quality, gap, numRoutes, eta, omega,
                perturbationType, insertionHeuristic, distanceLS, improvement));
            iterationWriter.flush();
        } catch (IOException e) {
            System.err.println("Error logging iteration: " + e.getMessage());
        }
    }
    
    public void logSummary(Long seed, String stoppingCriterion, double limit,
                          double finalQuality, double finalGap, double bestQuality, double bestGap,
                          int totalIterations, double timeToBest, int iterToBest, double totalTime) {
        try {
            String seedStr = (seed == null) ? "none" : seed.toString();
            summaryWriter.write(String.format("%s,%s,%s,%s,%.1f,%.2f,%.4f,%.2f,%.4f,%d,%.3f,%d,%.3f\n",
                runId, instanceName, seedStr, stoppingCriterion, limit,
                finalQuality, finalGap, bestQuality, bestGap,
                totalIterations, timeToBest, iterToBest, totalTime));
            summaryWriter.flush();
        } catch (IOException e) {
            System.err.println("Error logging summary: " + e.getMessage());
        }
    }
    
    public void close() {
        try {
            if (iterationWriter != null) iterationWriter.close();
            if (summaryWriter != null) summaryWriter.close();
            if (configWriter != null) configWriter.close();
        } catch (IOException e) {
            System.err.println("Error closing logger: " + e.getMessage());
        }
    }
    
    public String getRunDir() {
        return outputDir + "/" + instanceName + "/" + runId;
    }
    
    public String getRunId() {
        return runId;
    }
}
