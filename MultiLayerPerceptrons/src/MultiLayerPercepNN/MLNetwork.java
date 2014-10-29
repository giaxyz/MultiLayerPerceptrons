package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.HashMap;

public class MLNetwork {
	
	private ArrayList<HashMap<Integer,int[]>> networkStructure;
	private ArrayList<HashMap<Integer,double[]>> networkWeights;
	private HashMap<Integer, Double> biasWeights;
	private HashMap<Integer, Double> biasValues;
	private double learningRate;
	private double momentum;
	
	public MLNetwork(
			
			ArrayList<HashMap<Integer,int[]>> networkStructure, 
			ArrayList<HashMap<Integer,double[]>> networkWeights,
			HashMap<Integer, Double> biasWeights,
			HashMap<Integer, Double> biasValues,
			double learningRate,
			double momentum
			){
		
		this.networkStructure = networkStructure;
		this.networkWeights = networkWeights;
		this.biasWeights = biasWeights;
		this.biasValues = biasValues;
		this.learningRate = learningRate;
		this.momentum = momentum;
		
		getNetworkBiasWeights(false);
		getNetworkBiasValues(false);
		getNetworkWeights(false);
		getNetworkStructure(true);
		getNetworkLearningRate(true);
		getNetworkMomentum(true);
		
		boolean printInfo = true;
		boolean isOutput = false;
		boolean isInput = false;
		double biasValue = 1.0;
		double biasWeight = 0.0;
		int neuronID =  0;
		int layerID = 1; 
		Perceptron perceptron = createPerceptron(biasWeight, biasValue, neuronID, layerID, isOutput, isInput, printInfo);
	
	}
	
	// Create A Neuron
	
	public Perceptron createPerceptron(double biasWeight, double biasValue, int neuronID, int layerID, boolean isOutput, boolean isInput, boolean printInfo) {
		
		
		Perceptron perceptron = new Perceptron(biasWeight, biasValue, neuronID, layerID, isOutput, isInput);
		if(printInfo){
			
			System.out.println("Perceptron Details : " + perceptron);
		}
		
		return perceptron;
		
	}

	public HashMap<Integer, Double> getNetworkBiasWeights(boolean printInfo){
		
		if(printInfo){
			
			System.out.println("\nNetwork Bias weights : " );
			MlUtils.printHashMapIntDouble(this.biasWeights);
		}
		
		return this.biasWeights;
		
	}
	
	public HashMap<Integer, Double> getNetworkBiasValues(boolean printInfo){
		
		if(printInfo){
			
			System.out.println("\nNetwork Bias values : " );
			MlUtils.printHashMapIntDouble(this.biasValues);
		}
		
		return this.biasValues;
		
	}

	public ArrayList<HashMap<Integer,double[]>> getNetworkWeights(boolean printInfo){
		
		if(printInfo){
			
			for(int i = 0; i<this.networkWeights.size(); i++){
				
				HashMap<Integer,double[]> currentNeuronWeightInfo = networkWeights.get(i);
				System.out.println("\nNetwork Weights Layer " + (i-1) + " : ");
				MlUtils.printHashMapIntDoubleArray(currentNeuronWeightInfo);;
			}
			
		}
		
		return this.networkWeights;
	}

	public ArrayList<HashMap<Integer,int[]>> getNetworkStructure(boolean printInfo){
		
		if(printInfo){
			
			for(int i = 0; i<this.networkStructure.size(); i++){
				
				HashMap<Integer,int[]> currentNeuronWeightInfo = networkStructure.get(i);
				System.out.println("\nNetwork Connections Layer " + (i-1) + " : ");
				MlUtils.printHashMapIntIntArray(currentNeuronWeightInfo);;
			}
			
		}
		
		
		return this.networkStructure;
	}

	public double getNetworkLearningRate(boolean printInfo){
		
		if(printInfo){
			System.out.println("\nNetwork Learning Rate : " + this.learningRate);
		}
		return this.learningRate;
		
	}

	public double getNetworkMomentum(boolean printInfo){
		
		if(printInfo){
			System.out.println("\nCurrent Momentum : " + this.momentum);
		}
		return this.momentum;
		
	}
}
