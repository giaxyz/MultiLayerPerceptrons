package MultiLayerPercepNN;

import java.util.ArrayList;


public class Perceptron {
	


	private ArrayList<Double> deltaRow; // input weights

	private double output; // the calculated output
	private double biasWeight;
	private int layerID;
	private int neuronID;
	private boolean isOutput;
	private boolean isInput;
	private double sum;
	private String neuronDetails;
	private double biasValue;
	private double betaError;
	private int[] neuronConnections;
	private double[] neuronWeights;
	private ArrayList<ArrayList<Double>> inputData;
	private ArrayList<Double> inputs;
	
	
	
	public Perceptron(double biasWeight, double biasValue, 
			int neuronID, int layerID, 
			boolean isOutput, boolean isInput,
			int[] neuronConnections, double[] neuronWeights,
			ArrayList<ArrayList<Double>> inputData){
		
		this.biasWeight = biasWeight;
		this.biasValue = biasValue;
		this.isInput = isInput;
		this.isOutput = isOutput;
		this.layerID = layerID;
		this.neuronID = neuronID;
		this.neuronDetails = ("L_" + layerID + " N_" + neuronID );
		this.neuronConnections = neuronConnections;
		this.neuronWeights = neuronWeights;
		this.inputData = inputData;
		this.inputs = new ArrayList<Double>();
		
		this.sum = Double.NEGATIVE_INFINITY;
		this.betaError = Double.POSITIVE_INFINITY;
		this.output = Double.NEGATIVE_INFINITY; 
		this.deltaRow = new ArrayList<Double>(); 
	
		
	}
	
	// --------------   All Get and Set Methods  -------------------------------------
	
	private void setBiasWeight(double biasWeight){
		this.biasWeight = biasWeight;
	}

	private double getBiasWeight(boolean printInfo){
		
		if(printInfo){
			System.out.println("BiasWeight for " + this.neuronDetails + " : " + this.biasWeight);
		}
		return this.biasWeight;
	}
	
	private boolean getIsInput(boolean printInfo){
		
		if(printInfo){
			System.out.println(this.neuronDetails + " Input : " + this.isInput);
		}
		
		return this.isInput;
	}

	private boolean getIsOutput(boolean printInfo){
		
		if(printInfo){
			System.out.println(this.neuronDetails + " Output : " + this.isOutput);
		}
		
		return this.isOutput;
	}
	
	public void setSum(double sum){
		
		this.sum = sum;
	}
	
	public double getSum(boolean printInfo){
		
		if(printInfo){
			System.out.println("Sum for " + this.neuronDetails + " : " + this.sum);
		}
		
		return this.sum;
	}
	
	public int getLayerID(boolean printInfo){
		
		if(printInfo){
			
			System.out.println("Neuron Layer for " + this.neuronDetails + "  : " + this.layerID);
		}
		return this.layerID;
	}
	
	public int getNeuronID(boolean printInfo){
		
		if(printInfo){
			System.out.println("Neuron ID : " + this.neuronID);
		}
		return this.neuronID;
	}
	
	public String getNeuronDetails(boolean printInfo){
		if(printInfo){
			System.out.println("NeuronDetails : " + this.neuronDetails);
		}
		return this.neuronDetails;
	}
	
	public void setBiasValue(double biasValue){
		this.biasValue = biasValue;
	}
	
	public double getBiasValue(boolean printInfo){
		if(printInfo){
			System.out.println("BiasValue for " + this.neuronDetails + " : " + this.biasValue);
		}
		return this.biasValue;
	}

	public void setBetaError(double betaError){
		this.betaError = betaError;
	}
	
	public double getBetaError(boolean printInfo){
		
		if(printInfo){
			System.out.println("BetaError for " + this.neuronDetails + " : " + this.betaError);
		}
		return this.betaError;
	}
	
	public void setOutput(double output){
		this.output = output;
	}
	
	public double getOutput(boolean printInfo){
		
		if(printInfo){
			System.out.println("Output for " + this.neuronDetails + " : " + this.output);
		}
		return this.output;
		
	}
	
	public void setDeltaRow(ArrayList<Double> deltaRow){
		this.deltaRow = deltaRow;
	}
	
	public ArrayList<Double> getDeltaRow(boolean printInfo){
		
		if(printInfo){
			System.out.println("Delta Row for " + this.neuronDetails + " : " + this.deltaRow);
		}
		return this.deltaRow;
	}
	
	public String toString(){
		return this.neuronDetails;
	}
	

	public void setInputs(ArrayList<Double> inputs) {
		
		this.inputs = inputs;
		
	}

	public ArrayList<Double> getInputs(boolean printInfo) {
		
		if(printInfo){
			System.out.println("Inputs for Neuron " + neuronDetails + " : " + this.inputs);
		}
		return this.inputs;
		
	}

}
