package MultiLayerPercepNN;

import java.util.ArrayList;


public class Perceptron {
	
	private ArrayList<Double> inputValues; // Inputs from the previous node
	private ArrayList<Double> inputWeights; // input weights
	private ArrayList<Double> deltaRow; // input weights
	private ArrayList<Integer> inputIDs; // the id of the other neuron inputs
	private double output; // the calculated output
	private double bias;
	private double biasWeight;
	private int layerID;
	private int neuronID;
	private boolean isOutput;
	private boolean isInput;
	private double sum;
	private String neuronDetails;
	private double biasValue;
	private double betaError;
	
	
	public Perceptron(double biasWeight, double biasValue, int neuronID, int layerID, boolean isOutput, boolean isInput){
		
		this.biasWeight = biasWeight;
		this.biasValue = biasValue;
		this.isInput = isInput;
		this.isOutput = isOutput;
		this.sum = Double.NEGATIVE_INFINITY;
		this.inputIDs = new ArrayList<Integer>();
		this.inputValues = new ArrayList<Double>();
		this.layerID = layerID;
		this.neuronID = neuronID;
		this.neuronDetails = ("L_" + layerID + " N_" + neuronID );
		this.biasValue = biasValue;
		this.inputWeights = new ArrayList<Double>();
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
	
	private void setSum(double sum){
		
		this.sum = sum;
	}
	
	private double getSum(boolean printInfo){
		
		if(printInfo){
			System.out.println("Sum for " + this.neuronDetails + " : " + this.sum);
		}
		
		return this.sum;
	}
	
	public ArrayList<Integer> getInputIDs(boolean printInfo){
		
		if(printInfo){
			System.out.println("InputIDs for " + this.neuronDetails + " : " + this.inputIDs);
		}
		
		return this.inputIDs;
	}
	
	public ArrayList<Double> getInputValues(boolean printInfo){
		
		if(printInfo){
			System.out.println("Inputs for " + this.neuronDetails + " : " + this.inputValues);
		}
		return this.inputValues;
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
	
	public void setInputWeights(ArrayList<Double> inputWeights){
		this.inputWeights = inputWeights;
	}
	
	public ArrayList<Double> getInputWeights(boolean printInfo){
		
		if(printInfo){
			System.out.println("InputWeights for " + this.neuronDetails + " : " + this.inputWeights);
		}
		return this.inputWeights;
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

}
