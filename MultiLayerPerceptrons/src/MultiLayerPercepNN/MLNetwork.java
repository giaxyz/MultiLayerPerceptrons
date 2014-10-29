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
		
		boolean printInfo = true;
		boolean isOutput = false;
		boolean isInput = false;
		double biasValue = 1.0;
		double biasWeight = 0.0;
		int neuronID =  0;
		int layerID = 1; 
		Perceptron perceptron = createPerceptron(biasWeight, biasValue, neuronID, layerID, isOutput, isInput, printInfo);
		perceptron.getBetaError(false);
		perceptron.getInputIDs(false);
	}
	
	
	// Create A Perceptron
	public Perceptron createPerceptron(double biasWeight, double biasValue, int neuronID, int layerID, boolean isOutput, boolean isInput, boolean printInfo) {
		
		
		Perceptron perceptron = new Perceptron(biasWeight, biasValue, neuronID, layerID, isOutput, isInput);
		if(printInfo){
			
			System.out.println("Perceptron Details : " + perceptron);
		}
		return perceptron;
		
	}
}
