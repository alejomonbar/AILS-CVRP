package SearchMethod;

import java.util.Random;

import Data.Instance;
import ML.GNNConstructionHeuristic;
import Solution.Node;
import Solution.Route;
import Solution.Solution;


public class ConstructSolution 
{
	private Route routes[];
	private double f=0;
	private int numRoutes;
	private Node []solution;
	protected Random rand;
	protected int size;
	Instance instance;
	Node notInserted[];
	int countNotInserted=0;
	
	// GNN-based construction
	private GNNConstructionHeuristic gnnHeuristic;
	private boolean useGNN;
	
	public ConstructSolution(Instance instance,Config config)
	{
		// Initialize random with seed if provided
		if(config.getRandomSeed() != null) {
			this.rand = new Random(config.getRandomSeed());
		} else {
			this.rand = new Random();
		}
		this.instance=instance;
		this.routes=new Route[instance.getMaxNumberRoutes()];
		this.size=instance.getSize()-1;
		this.notInserted=new Node[size];
		
		// Initialize GNN if enabled
		this.useGNN = config.isUseGNN();
		if (this.useGNN) {
			this.gnnHeuristic = new GNNConstructionHeuristic(config.getGnnModelPath());
			if (!this.gnnHeuristic.isAvailable()) {
				System.err.println("GNN initialization failed. Falling back to classical heuristic.");
				this.useGNN = false;
			}
		}
	}
	
	private void setSolution(Solution solution) 
	{
		this.numRoutes=solution.numRoutes;
		this.solution=solution.getSolution();
		this.f=solution.f;
		for (int i = 0; i < routes.length; i++) 
			this.routes[i]=solution.routes[i];
	}

	private void assignResult(Solution solution) 
	{
		solution.numRoutes=this.numRoutes;
		solution.f=this.f;
		for (int i = 0; i < routes.length; i++) 
			solution.routes[i]=this.routes[i];
	}

	public void construct(Solution s)
	{
		// Try GNN construction first if enabled
		if (useGNN && gnnHeuristic != null && gnnHeuristic.isAvailable()) {
			boolean success = constructWithGNN(s);
			if (success) {
				return; // GNN construction successful
			}
			// Fall through to classical construction if GNN fails
			System.err.println("GNN construction failed. Using classical heuristic.");
		}
		
		// Classical random construction
		constructClassical(s);
	}
	
	/**
	 * GNN-based construction
	 */
	private boolean constructWithGNN(Solution s) {
		try {
			setSolution(s);
			
			// Get tour from GNN
			int[] tour = gnnHeuristic.constructSolution(instance);
			if (tour == null || tour.length == 0) {
				return false;
			}
			
			// Clean routes
			for (int i = 0; i < routes.length; i++)
				routes[i].clean();
			
			f = 0;
			
			// Convert GNN tour to routes (tour is a sequence of customer visits)
			// We need to split it into routes respecting capacity constraints
			int currentRoute = 0;
			double currentLoad = 0;
			boolean routeStarted = false;
			
			for (int i = 0; i < tour.length; i++) {
				int nodeIndex = tour[i];
				
				// Skip depot (0)
				if (nodeIndex == 0) {
					if (routeStarted) {
						// Depot means end of current route
						currentRoute++;
						currentLoad = 0;
						routeStarted = false;
						if (currentRoute >= routes.length) {
							break; // No more routes available
						}
					}
					continue;
				}
				
				Node node = solution[nodeIndex - 1]; // solution array doesn't include depot
				
				// Check capacity
				if (currentLoad + node.demand <= instance.getCapacity()) {
					// Add to current route
					double cost = routes[currentRoute].addNodeEndRoute(node);
					f += cost;
					currentLoad += node.demand;
					routeStarted = true;
				} else {
					// Start new route
					currentRoute++;
					if (currentRoute >= routes.length) {
						System.err.println("GNN tour exceeds available routes");
						return false;
					}
					double cost = routes[currentRoute].addNodeEndRoute(node);
					f += cost;
					currentLoad = node.demand;
					routeStarted = true;
				}
			}
			
			// Count used routes
			numRoutes = 0;
			for (int i = 0; i < routes.length; i++) {
				if (routes[i].getNumElements() > 0) {
					numRoutes++;
				}
			}
			
			assignResult(s);
			s.removeEmptyRoutes();
			
			System.out.println("GNN construction successful. Routes: " + numRoutes + ", Cost: " + f);
			return true;
			
		} catch (Exception e) {
			System.err.println("GNN construction error: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
	}
	
	/**
	 * Classical random construction (original implementation)
	 */
	private void constructClassical(Solution s)
	{
		setSolution(s);
		
		for (int i = 0; i < routes.length; i++)
			routes[i].clean();
		
		int index;
		Node node,bestNode;
		f=0;
		countNotInserted=0;
		
		for (int i = 0; i < size; i++) 
			notInserted[countNotInserted++]=solution[i];
		
		for (int i = 0; i < numRoutes; i++)
		{
			index=rand.nextInt(countNotInserted);
			f+=routes[i].addNodeEndRoute(notInserted[index]);
			
			node=notInserted[index];
			notInserted[index]=notInserted[countNotInserted-1];
			notInserted[--countNotInserted]=node;
		}
		
		while(countNotInserted>0) 
		{
			index=rand.nextInt(countNotInserted);
			node=notInserted[index];
			bestNode=getBestNoRoutes(node);
			f+=bestNode.route.addAfter(node, bestNode);
			notInserted[index]=notInserted[countNotInserted-1];
			notInserted[--countNotInserted]=node;
		}
		
		assignResult(s);
		s.removeEmptyRoutes();
	}
	
	protected Node getBestKNNNo(Node no)
	{
		double bestCost=Double.MAX_VALUE;
		Node aux,bestNode=null;
		double cost,costPrev;
		
		for (int i = 0; i < solution.length; i++) 
		{
			aux=solution[i];
			if(aux.nodeBelong)
			{
				cost=instance.dist(aux.name,no.name)+instance.dist(no.name,aux.next.name)-instance.dist(aux.name,aux.next.name);
				if(cost<bestCost)
				{
					bestCost=cost;
					bestNode=aux;
				}
			}
		}
		cost=instance.dist(bestNode.name,no.name)+instance.dist(no.name,bestNode.next.name)-instance.dist(bestNode.name,bestNode.next.name);
		costPrev=instance.dist(bestNode.prev.name,no.name)+instance.dist(no.name,bestNode.name)-instance.dist(bestNode.prev.name,bestNode.name);
		if(cost<costPrev)
			return bestNode;
		
		return bestNode.prev;
	}
	
	protected Node getBestNoRoutes(Node no)
	{
		double bestCost=Double.MAX_VALUE;
		Node aux,bestNode=null;
		
		for (int i = 0; i < numRoutes; i++) 
		{
			aux=routes[i].findBestPosition(no);
			if(routes[i].lowestCost<bestCost)
			{
				bestCost=routes[i].lowestCost;
				bestNode=aux;
			}
		}
		
		return bestNode;
	}
	
	/**
	 * Cleanup GNN resources
	 */
	public void cleanup() {
		if (gnnHeuristic != null) {
			gnnHeuristic.close();
		}
	}
	
}

	
