package SearchMethod;

import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Random;

import Auxiliary.Distance;
import Data.Instance;
import DiversityControl.DistAdjustment;
import DiversityControl.OmegaAdjustment;
import DiversityControl.AcceptanceCriterion;
import DiversityControl.IdealDist;
import Improvement.LocalSearch;
import Improvement.IntraLocalSearch;
import Improvement.FeasibilityPhase;
import Perturbation.InsertionHeuristic;
import Perturbation.Perturbation;
import Solution.Solution;
import RL.OperatorSelector;
import RL.OmegaController;

public class AILSII 
{
	//----------Problema------------
	Solution solution,referenceSolution,bestSolution;
	
	Instance instance;
	Distance pairwiseDistance;
	double bestF=Double.MAX_VALUE;
	double executionMaximumLimit;
	double optimal;
	
	//----------caculoLimiar------------
	int numIterUpdate;

	//----------Metricas------------
	int iterator,iteratorMF;
	long first,ini;
	double timeAF,totalTime,time;
	
	Random rand;
	
	HashMap<String,OmegaAdjustment>omegaSetup=new HashMap<String,OmegaAdjustment>();

	double distanceLS;
	
	Perturbation[] pertubOperators;
	Perturbation selectedPerturbation;
	OperatorSelector operatorSelector; // RL-based operator selection
	OmegaController omegaController; // RL-based omega control
	
	FeasibilityPhase feasibilityOperator;
	ConstructSolution constructSolution;
	
	LocalSearch localSearch;

	InsertionHeuristic insertionHeuristic;
	IntraLocalSearch intraLocalSearch;
	AcceptanceCriterion acceptanceCriterion;
//	----------Mare------------
	DistAdjustment distAdjustment;
//	---------Print----------
	boolean print=true;
	IdealDist idealDist;
	
	double epsilon;
	DecimalFormat deci=new DecimalFormat("0.0000");
	StoppingCriterionType stoppingCriterionType;
	ResultsLogger logger;
	InputParameters inputParams;
	double previousBestF=Double.MAX_VALUE;
	
	public AILSII(Instance instance,InputParameters reader)
	{ 
		this.instance=instance;
		this.inputParams=reader;
		Config config=reader.getConfig();
		
		// Initialize random number generator with seed if provided
		if(config.getRandomSeed() != null) {
			this.rand = new Random(config.getRandomSeed());
			System.out.println("Using random seed: " + config.getRandomSeed());
		} else {
			this.rand = new Random();
			System.out.println("Using non-deterministic random (no seed)");
		}
		
		// Initialize results logger
		this.logger = new ResultsLogger("experiments", reader.getFile(), config);
		System.out.println("Logging results to: " + logger.getRunDir());
		
		this.optimal=reader.getBest();
		this.executionMaximumLimit=reader.getTimeLimit();
		
		this.epsilon=config.getEpsilon();
		this.stoppingCriterionType=config.getStoppingCriterionType();
		this.idealDist=new IdealDist();
		this.solution =new Solution(instance,config);
		this.referenceSolution =new Solution(instance,config);
		this.bestSolution =new Solution(instance,config);
		this.numIterUpdate=config.getGamma();
		
		this.pairwiseDistance=new Distance();
		
		this.pertubOperators=new Perturbation[config.getPerturbation().length];
		
		this.distAdjustment=new DistAdjustment( idealDist, config, executionMaximumLimit);
		
		this.intraLocalSearch=new IntraLocalSearch(instance,config);
		
		this.localSearch=new LocalSearch(instance,config,intraLocalSearch);
		
		this.feasibilityOperator=new FeasibilityPhase(instance,config,intraLocalSearch);
		
		this.constructSolution=new ConstructSolution(instance,config);
		
		OmegaAdjustment newOmegaAdjustment;
		for (int i = 0; i < config.getPerturbation().length; i++) 
		{
			newOmegaAdjustment=new OmegaAdjustment(config.getPerturbation()[i], config,instance.getSize(),idealDist);
			omegaSetup.put(config.getPerturbation()[i]+"", newOmegaAdjustment);
		}
		
		this.acceptanceCriterion=new AcceptanceCriterion(instance,config,executionMaximumLimit);

		try 
		{
			for (int i = 0; i < pertubOperators.length; i++) 
			{
				this.pertubOperators[i]=(Perturbation) Class.forName("Perturbation."+config.getPerturbation()[i]).
				getConstructor(Instance.class,Config.class,HashMap.class,IntraLocalSearch.class).
				newInstance(instance,config,omegaSetup,intraLocalSearch);
			}
			
		} catch (InstantiationException | IllegalAccessException | IllegalArgumentException
				| InvocationTargetException | NoSuchMethodException | SecurityException
				| ClassNotFoundException e) {
			e.printStackTrace();
		}
		
		// Initialize RL-based operator selector
		this.operatorSelector = new OperatorSelector(
			pertubOperators, 
			config.getUcbExplorationParam(), 
			config.getRandomSeed(), 
			config.isRlOperatorSelection()
		);
		System.out.println("RL operator selection: " + (config.isRlOperatorSelection() ? "ENABLED" : "DISABLED"));
		
		// Initialize RL-based omega controller
		this.omegaController = new OmegaController(
			config.getDMin(), // Use dMin/dMax as omega bounds
			config.getDMax(),
			config.getQLearningRate(),
			config.getQDiscountFactor(),
			config.getQEpsilon(),
			config.getRandomSeed(),
			config.isRlOmegaControl()
		);
		System.out.println("RL omega control: " + (config.isRlOmegaControl() ? "ENABLED" : "DISABLED"));
		
	}

	public void search()
	{
		iterator=0;
		first=System.currentTimeMillis();
		referenceSolution.numRoutes=instance.getMinNumberRoutes();
		constructSolution.construct(referenceSolution);
		
		feasibilityOperator.makeFeasible(referenceSolution);
		localSearch.localSearch(referenceSolution,true);
		bestSolution.clone(referenceSolution);
		while(!stoppingCriterion())
		{
			iterator++;

			solution.clone(referenceSolution);
			
			double previousQuality = referenceSolution.f;
			double previousBestQuality = bestF;
			
			// Use RL-based operator selection or random selection
			selectedPerturbation = operatorSelector.selectOperator();
			
			// RL-based omega control: Get current omega before perturbation
			int iterationsSinceImprovement = iterator - iteratorMF;
			
			// Apply perturbation first (this sets chosenOmega)
			selectedPerturbation.applyPerturbation(solution);
			
			// Now we can get and adjust omega
			double currentOmega = selectedPerturbation.omega; // Use the omega that was just applied
			int omegaAction = omegaController.selectAction(iterationsSinceImprovement, currentOmega);
			// Store action for logging, but omega for next iteration will be adjusted later via distAdjustment
			
			feasibilityOperator.makeFeasible(solution);
			localSearch.localSearch(solution,true);
			distanceLS=pairwiseDistance.pairwiseSolutionDistance(solution,referenceSolution);
			
			evaluateSolution();
			
			// Calculate rewards and update RL agents
			boolean foundNewBest = bestF < previousBestQuality;
			
			// Operator selector reward
			double operatorReward = OperatorSelector.calculateImprovementReward(previousQuality, solution.f);
			operatorSelector.giveReward(operatorReward);
			
			// Omega controller reward and update
			double omegaReward = OmegaController.calculateReward(previousQuality, solution.f, foundNewBest);
			int newIterationsSinceImprovement = iterator - iteratorMF;
			double finalOmega = selectedPerturbation.omega;
			omegaController.updateQ(omegaReward, newIterationsSinceImprovement, finalOmega);
			
			// Standard diversity control adjustment
			distAdjustment.distAdjustment();
			
			// Apply RL-based omega adjustment (overrides or modifies distAdjustment if enabled)
			if (omegaController.isEnabled()) {
				double adjustedOmega = omegaController.applyAction(
					selectedPerturbation.getChosenOmega().getActualOmega(), 
					omegaAction
				);
				selectedPerturbation.getChosenOmega().setActualOmega(adjustedOmega);
			}
			
			selectedPerturbation.getChosenOmega().setDistance(distanceLS);//update
			
			if(acceptanceCriterion.acceptSolution(solution))
				referenceSolution.clone(solution);
		}
		
		totalTime=(double)(System.currentTimeMillis()-first)/1000;
		
		// Print operator selection statistics
		if(inputParams.getConfig().isRlOperatorSelection()) {
			System.out.println(operatorSelector.getAllStats());
		}
		
		// Print omega control statistics
		if(inputParams.getConfig().isRlOmegaControl()) {
			System.out.println(omegaController.getStats());
		}
		
		// Log final summary
		Config config = inputParams.getConfig();
		logger.logSummary(
			config.getRandomSeed(),
			stoppingCriterionType.toString(),
			executionMaximumLimit,
			solution.f,
			getGap(),
			bestF,
			100*((bestF-optimal)/optimal),
			iterator,
			timeAF,
			iteratorMF,
			totalTime
		);
		
		// Close logger
		logger.close();
		System.out.println("\nResults saved to: " + logger.getRunDir());
	}
	
	public void evaluateSolution()
	{
		if((solution.f-bestF)<-epsilon)
		{		
			bestF=solution.f;
			
			bestSolution.clone(solution);
			iteratorMF=iterator;
			timeAF=(double)(System.currentTimeMillis()-first)/1000;
				
			if(print)
			{
				System.out.println("solution quality: "+bestF
				+" gap: "+deci.format(getGap())+"%"
				+" K: "+solution.numRoutes
				+" iteration: "+iterator
				+" eta: "+deci.format(acceptanceCriterion.getEta())
				+" omega: "+deci.format(selectedPerturbation.omega)
				+" time: "+timeAF
				);
			}
		}
		
		// Log every iteration with detailed metrics (after potential best update)
		double improvement = (previousBestF == Double.MAX_VALUE) ? 0 : previousBestF - bestF;
		previousBestF = bestF;
		
		double currentTime = (double)(System.currentTimeMillis()-first)/1000;
		logger.logIteration(
			iterator,
			currentTime,
			solution.f,
			getGap(),
			solution.numRoutes,
			acceptanceCriterion.getEta(),
			selectedPerturbation.omega,
			selectedPerturbation.getPerturbationType().toString(),
			selectedPerturbation.selectedInsertionHeuristic.toString(),
			distanceLS,
			improvement,
			operatorSelector.getLastSelectedIndex(),
			operatorSelector.getLastReward(),
			omegaController.getLastAction()
		);
	}
	
	private boolean stoppingCriterion()
	{
		switch(stoppingCriterionType)
		{
			case Iteration: 	if(bestF<=optimal||executionMaximumLimit<=iterator)
									return true;
								break;
							
			case Time: 	if(bestF<=optimal||executionMaximumLimit<(System.currentTimeMillis()-first)/1000)
							return true;
						break;
		}
		return false;
	}
	
	public static void main(String[] args) 
	{
		InputParameters reader=new InputParameters();
		reader.readingInput(args);
		
		Instance instance=new Instance(reader);
		
		AILSII ailsII=new AILSII(instance,reader);
		
		ailsII.search();
	}
	
	public Solution getBestSolution() {
		return bestSolution;
	}

	public double getBestF() {
		return bestF;
	}

	public double getGap()
	{
		return 100*((bestF-optimal)/optimal);
	}
	
	public boolean isPrint() {
		return print;
	}

	public void setPrint(boolean print) {
		this.print = print;
	}

	public Solution getSolution() {
		return solution;
	}

	public int getIterator() {
		return iterator;
	}

	public String printOmegas()
	{
		String str="";
		for (int i = 0; i < pertubOperators.length; i++) 
		{
			str+="\n"+omegaSetup.get(this.pertubOperators[i].perturbationType+""+referenceSolution.numRoutes);
		}
		return str;
	}
	
	public Perturbation[] getPertubOperators() {
		return pertubOperators;
	}
	
	public double getTotalTime() {
		return totalTime;
	}
	
	public double getTimePerIteration() 
	{
		return totalTime/iterator;
	}

	public double getTimeAF() {
		return timeAF;
	}

	public int getIteratorMF() {
		return iteratorMF;
	}
	
	public double getConvergenceIteration()
	{
		return (double)iteratorMF/iterator;
	}
	
	public double convergenceTime()
	{
		return (double)timeAF/totalTime;
	}
	
}
