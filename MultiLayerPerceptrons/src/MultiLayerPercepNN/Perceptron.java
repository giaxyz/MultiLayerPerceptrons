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
	
	
	public void setDeltaRow(ArrayList<Double> deltaValues, boolean printInfo){
		
		this.deltaRow = deltaValues;
		if(printInfo){
			System.out.println("Delta Values set to : " + deltaValues);
		}
		
		
		
	}
	
	private void setBiasWeight(double biasWeight){
		this.biasWeight = biasWeight;
	}

	public double getBiasWeight(boolean printInfo){
		
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
	

	public double computeSum(boolean printInfo) throws Exception {
		
		if(printInfo){
			
			System.out.println("\n-------Computing Sum For " + neuronDetails);
		}
		
		
		double sum = getSum(false);
		
		if(layerID == -1){
			if(this.sum == Double.NEGATIVE_INFINITY){
				throw new Exception("The inputs for the dummy input neurons not set properly");
			}
		}
		
		if((inputs.isEmpty()) && this.layerID != -1){
			throw new Exception("The input array for N" + neuronID + "_L" + layerID + " is Empty");
		}
		
		if(printInfo){
			System.out.println("\t Inputs are : " + inputs);
			System.out.print("\t Weights are : ");
			MlUtils.printDoubleArray(neuronWeights);
			System.out.print("\n");
		}
		
		
		
		if(layerID != -1){
			
			if(printInfo){
				System.out.println("Current neuron " + neuronDetails);
				System.out.println("Inputs are " + inputs);
				for(int i = 0; i< neuronWeights.length; i++){
					System.out.println(" W " + neuronWeights[i]);
				}
			}
			
			
			sum = 0.0;
			if(inputs.size() != neuronWeights.length){
				throw new Exception("N" + neuronID + "_L" + layerID + " weights and inputs matrices do not match");
			}
			
			
			
			if(printInfo){
				System.out.println("\t Sum for L_" + layerID + "_N_" + neuronID);
			}
			
			for(int i = 0; i< inputs.size(); i++){
				
				double inputVal = inputs.get(i);
				double weightVal = neuronWeights[i];
				sum += (inputVal * weightVal);
				
				
				if(printInfo){
					System.out.println("\t\t Input : " + inputVal + " Weight " + weightVal);
				}
				
			}
			
			if(printInfo){
				
				System.out.println("\tFinal sum " + sum);
			}
			
			
		}
		
		
		this.sum = sum;
		return sum;
		
	}

	public double activateSigmoid(double sumValue, boolean printInfo){
		
		double finalSigmoidVal;
		
		if(!(isInput)){
			
			double thresholdVal =  1.0 / (1 + Math.exp(-1.0 * sumValue));
			thresholdVal = MlUtils.formatDouble(thresholdVal);
			
			finalSigmoidVal = thresholdVal;
		}else{
			finalSigmoidVal = sumValue;
		}
			
		setOutput(finalSigmoidVal); // set the output value, if it's not an input neuron
		
		if(printInfo){
			
		System.out.println("\t\tActivating " + neuronDetails +  " " +  sumValue);
			System.out.println("\t\tThreshold : " + finalSigmoidVal);
			if(isInput){
				System.out.print("\t\t\tno activation because it's an input dummy neuron");
			}
		}
		
		return finalSigmoidVal;
		
	}
	
	public double setBetaError(double betaError, boolean printInfo){
		
		this.betaError = betaError;
		
		if(printInfo){
			System.out.println("\t" + neuronDetails + " betaError new value " + this.betaError);
		}
		
		return this.betaError;
		
	}
	
	public boolean getIsInput(){
		return this.isInput;
	}
	
	public boolean getIsOutput(){
		return this.isOutput;
	}
	
	public void troubleshoot(){
		
		System.out.println(" \n-----Troubleshooting ---: Weights, " + this.neuronDetails);
		for(int i = 0; i< this.neuronWeights.length; i++){
			System.out.println(neuronWeights[i]);
		}
		
		System.out.println(" \n-----Troubleshooting ---: INputs, " + this.neuronDetails);
		for(int i = 0; i< this.inputs.size(); i++){
			System.out.println(inputs.get(i));
		}
	}

	public double[] getInputWeightsRow(boolean printInfo) {
		
		if(printInfo){
			System.out.println("WeightsRow for Neuron " + neuronID + " in Layer " + layerID + " is : ");
			for(int i = 0; i< this.neuronWeights.length; i++){
				System.out.println(" Weight " + i + " is " + this.neuronWeights[i]);
			}
		}
		return this.neuronWeights;
	
	}

	public int[] getNeuronConnections(boolean printInfo){
		
		if(printInfo){
			System.out.println("Connections for Neuron " + neuronID + " in Layer " + layerID + " is : ");
			for(int i = 0; i< this.neuronConnections.length; i++){
				System.out.println(" Connection # " + i + " is " + this.neuronConnections[i]);
			}
		}
		return this.neuronConnections;
	}
	
}
