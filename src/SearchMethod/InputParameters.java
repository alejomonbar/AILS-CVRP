package SearchMethod;

import java.io.File;
import java.util.Arrays;

public class InputParameters 
{
	
	private String file="";
	private boolean rounded=true;
	private double limit=Double.MAX_VALUE;
	private double best=0;
	private Config config =new Config();
	
	public void readingInput(String[] args)
	{
		try 
		{
			for (int i = 0; i < args.length-1; i+=2) 
			{
				switch(args[i])
				{
					case "-file": file=getAddress(args[i+1]);break;
					case "-rounded": rounded=getRound(args[i+1]);break;
					case "-limit": limit=getLimit(args[i+1]);break;
					case "-best": best=getBest(args[i+1]);break;
					case "-stoppingCriterion": config.setStoppingCriterionType(getStoppingCriterion(args[i+1]));break;
					case "-dMax": config.setDMax(getDMax(args[i+1]));break;
					case "-dMin": config.setDMin(getDMin(args[i+1]));break;
					case "-gamma": config.setGamma(getGamma(args[i+1]));break;
					case "-varphi": config.setVarphi(getVarphi(args[i+1]));break;
					case "-seed": config.setRandomSeed(getSeed(args[i+1]));break;
					case "-useRL": config.setRlOperatorSelection(getUseRL(args[i+1]));break;
					case "-ucbExploration": config.setUcbExplorationParam(getUcbExploration(args[i+1]));break;
					case "-useOmegaRL": config.setRlOmegaControl(getUseOmegaRL(args[i+1]));break;
					case "-qAlpha": config.setQLearningRate(getQLearningRate(args[i+1]));break;
					case "-qGamma": config.setQDiscountFactor(getQDiscountFactor(args[i+1]));break;
					case "-qEpsilon": config.setQEpsilon(getQEpsilon(args[i+1]));break;
					case "-useVarphiRL": config.setRlVarphiControl(getUseVarphiRL(args[i+1]));break;
					case "-varphiQAlpha": config.setVarphiQLearningRate(getVarphiQLearningRate(args[i+1]));break;
					case "-varphiQGamma": config.setVarphiQDiscountFactor(getVarphiQDiscountFactor(args[i+1]));break;
					case "-varphiQEpsilon": config.setVarphiQEpsilon(getVarphiQEpsilon(args[i+1]));break;
					
				}
			}
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		
		System.out.println("File: "+file);
		System.out.println("Rounded: "+rounded);
		System.out.println("limit: "+limit);
		System.out.println("Best: "+best);
		System.out.println("LimitTime: "+limit);
		System.out.println(config);
	}
	
	
	public String getAddress(String text)
	{
		try 
		{
			File file=new File(text);
			if(file.exists()&&!file.isDirectory())
				return text;
			else
				System.err.println("The -file parameter must contain the address of a valid file.");
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		return "";	
	}
	
	public boolean getRound(String text)
	{
		rounded=true;
		try 
		{
			if(text.equals("false")||text.equals("true"))
				rounded=Boolean.valueOf(text);
			else
				System.err.println("The -rounded parameter must have the values false or true.");
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		return rounded;
	}
	
	public double getLimit(String text)
	{
		try 
		{
			limit=Double.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) {
			System.err.println("The -limit parameter must contain a valid real value.");
		}
		return limit;
	}
	
	public double getBest(String text)
	{
		try 
		{
			best=Double.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) {
			System.err.println("The -best parameter must contain a valid real value.");
		}
		return best;
	}
	
	public int getVarphi(String text)
	{
		int varphi=40;
		try 
		{
			varphi=Integer.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) {
			System.err.println("The -varphi parameter must contain a valid integer value.");
		}
		return varphi;
	}
	
	public int getGamma(String text)
	{
		int gamma=30;
		try 
		{
			gamma=Integer.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) {
			System.err.println("The -gamma parameter must contain a valid integer value.");
		}
		return gamma;
	}
	
	public int getDMax(String text)
	{
		int dMax=30;
		try 
		{
			dMax=Integer.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) {
			System.err.println("The -dMax parameter must contain a valid integer value.");
		}
		return dMax;
	}
	
	public int getDMin(String text)
	{
		int dMin=15;
		try 
		{
			dMin=Integer.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) {
			System.err.println("The -dMin parameter must contain a valid integer value.");
		}
		return dMin;
	}
	
	public StoppingCriterionType getStoppingCriterion(String text)
	{
		StoppingCriterionType stoppingCriterion=StoppingCriterionType.Time;
		try 
		{
			stoppingCriterion=StoppingCriterionType.valueOf(text);
		} 
		catch (java.lang.IllegalArgumentException e) 
		{
			System.err.println("The -stoppingCriterion parameter must have the values "+Arrays.toString(StoppingCriterionType.values())+".");
		}
		return stoppingCriterion;
	}

	public Long getSeed(String text)
	{
		Long seed = null;
		try 
		{
			seed = Long.valueOf(text);
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -seed parameter must contain a valid long integer value.");
		}
		return seed;
	}

	public boolean getUseRL(String text)
	{
		boolean useRL = false;
		try 
		{
			useRL = Boolean.parseBoolean(text);
		} 
		catch (Exception e) 
		{
			System.err.println("The -useRL parameter must be 'true' or 'false'.");
		}
		return useRL;
	}

	public double getUcbExploration(String text)
	{
		double ucbExploration = Math.sqrt(2);
		try 
		{
			ucbExploration = Double.parseDouble(text);
			if (ucbExploration <= 0) {
				System.err.println("The -ucbExploration parameter must be positive. Using default: " + Math.sqrt(2));
				ucbExploration = Math.sqrt(2);
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -ucbExploration parameter must contain a valid double value.");
		}
		return ucbExploration;
	}

	public boolean getUseOmegaRL(String text)
	{
		boolean useOmegaRL = false;
		try 
		{
			useOmegaRL = Boolean.parseBoolean(text);
		} 
		catch (Exception e) 
		{
			System.err.println("The -useOmegaRL parameter must be 'true' or 'false'.");
		}
		return useOmegaRL;
	}

	public double getQLearningRate(String text)
	{
		double qAlpha = 0.1;
		try 
		{
			qAlpha = Double.parseDouble(text);
			if (qAlpha <= 0 || qAlpha > 1) {
				System.err.println("The -qAlpha parameter must be in (0,1]. Using default: 0.1");
				qAlpha = 0.1;
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -qAlpha parameter must contain a valid double value.");
		}
		return qAlpha;
	}

	public double getQDiscountFactor(String text)
	{
		double qGamma = 0.9;
		try 
		{
			qGamma = Double.parseDouble(text);
			if (qGamma < 0 || qGamma > 1) {
				System.err.println("The -qGamma parameter must be in [0,1]. Using default: 0.9");
				qGamma = 0.9;
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -qGamma parameter must contain a valid double value.");
		}
		return qGamma;
	}

	public double getQEpsilon(String text)
	{
		double qEpsilon = 0.1;
		try 
		{
			qEpsilon = Double.parseDouble(text);
			if (qEpsilon < 0 || qEpsilon > 1) {
				System.err.println("The -qEpsilon parameter must be in [0,1]. Using default: 0.1");
				qEpsilon = 0.1;
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -qEpsilon parameter must contain a valid double value.");
		}
		return qEpsilon;
	}

	public boolean getUseVarphiRL(String text)
	{
		boolean useVarphiRL = false;
		try 
		{
			useVarphiRL = Boolean.parseBoolean(text);
		} 
		catch (Exception e) 
		{
			System.err.println("The -useVarphiRL parameter must be 'true' or 'false'.");
		}
		return useVarphiRL;
	}

	public double getVarphiQLearningRate(String text)
	{
		double varphiQAlpha = 0.1;
		try 
		{
			varphiQAlpha = Double.parseDouble(text);
			if (varphiQAlpha <= 0 || varphiQAlpha > 1) {
				System.err.println("The -varphiQAlpha parameter must be in (0,1]. Using default: 0.1");
				varphiQAlpha = 0.1;
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -varphiQAlpha parameter must contain a valid double value.");
		}
		return varphiQAlpha;
	}

	public double getVarphiQDiscountFactor(String text)
	{
		double varphiQGamma = 0.9;
		try 
		{
			varphiQGamma = Double.parseDouble(text);
			if (varphiQGamma < 0 || varphiQGamma > 1) {
				System.err.println("The -varphiQGamma parameter must be in [0,1]. Using default: 0.9");
				varphiQGamma = 0.9;
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -varphiQGamma parameter must contain a valid double value.");
		}
		return varphiQGamma;
	}

	public double getVarphiQEpsilon(String text)
	{
		double varphiQEpsilon = 0.1;
		try 
		{
			varphiQEpsilon = Double.parseDouble(text);
			if (varphiQEpsilon < 0 || varphiQEpsilon > 1) {
				System.err.println("The -varphiQEpsilon parameter must be in [0,1]. Using default: 0.1");
				varphiQEpsilon = 0.1;
			}
		} 
		catch (java.lang.NumberFormatException e) 
		{
			System.err.println("The -varphiQEpsilon parameter must contain a valid double value.");
		}
		return varphiQEpsilon;
	}

	public String getFile() {
		return file;
	}

	public boolean isRounded() {
		return rounded;
	}

	public double getTimeLimit() {
		return limit;
	}

	public double getBest() {
		return best;
	}


	public Config getConfig() {
		return config;
	}
	
}
