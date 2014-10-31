package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.HashMap;


public class Backpropagator {

	private MLNetwork network;
	private int networkSize;
	private boolean test1Neuron;
	private int neuronToTest;
	private int currentExample;
	private boolean printBackPropInfo;
	
	public Backpropagator(MLNetwork network) {
		
		this.network = network;
		this.networkSize = network.getNetworkBiasValues(false).size();
		this.test1Neuron = true; // set this to test only 1 layer, it will backprop from the out to the test neuron
		this.neuronToTest = 3;
		this.currentExample = 0;
		this.printBackPropInfo = true;
		
	}

	public void backpropagate(int exampleNumber) {
		
		System.out.println("\n\n\t\t ----------------  Backpropagating ---------------  ");
		
		// Make sure that the last neuron is first set to the last actual neuron, not an input one
		int lastActualNeuron = getLastNeuronInNetwork(false);
		int lastNeuron = confirmLastNeuron(lastActualNeuron, false);
		
		//Set the current Example
		setCurrentExample(exampleNumber);
		
		// Now backpropagate from the last neuron to the first
		for(int i = (this.networkSize - 1);  i>=lastNeuron; i--){
		
			Perceptron currentNeuron = this.network.getNeurons(false).get(i);
			
			if(printBackPropInfo){
				System.out.println("\n\n\n-- Neuron " + currentNeuron + "-- : ");
			}
			
			// compute the Beta Error for the Neuron
			double betaError = computeBetaError(currentNeuron, false);
			currentNeuron.setBetaError(betaError, false);
			
			if(printBackPropInfo){
				System.out.println("\t---" +  " betaError set to : " + betaError + "\n");
			}
			
			// Compute the delta values for the neuron
			ArrayList<Double> deltaValues = computeDeltaValues(currentNeuron, false);
			currentNeuron.setDeltaRow(deltaValues, false);
			
			if(printBackPropInfo){
				System.out.println("\t---" + " deltas set to : " + deltaValues);
				
			}
			
		}
		
	}
	

	
	private ArrayList<Double> computeDeltaValues(Perceptron currentNeuron,
			boolean printInfo) {
		
		ArrayList<Double> deltaValues = new ArrayList<Double>();
		double learningRate = network.getLearningRate(false);
		double betaValue = currentNeuron.getBetaError(false);
		ArrayList<Double> previousLayerOutputs = getPreviousLayerOutput(currentNeuron, false);
	
		
		
		for(int i = 0; i< previousLayerOutputs.size(); i++){
			
			if(printBackPropInfo){
				System.out.println("\t---Computing for " + currentNeuron + "(" + learningRate + ") * (" + betaValue + ")" + " * (" + previousLayerOutputs.get(i) + ")\n");
			}
			
			double deltaValue = MlUtils.formatDouble((learningRate)*(betaValue)*(previousLayerOutputs.get(i)));
			deltaValues.add(deltaValue);
		}
		
		return deltaValues;

	}
	
	public ArrayList<Double> getPreviousLayerOutput(Perceptron neuron, boolean printInfo){
		
		ArrayList<Double> previousLayerOutputs = new ArrayList<Double>();
		int currentNeuronID = neuron.getNeuronID(false);
		int currentNeuronLayerID = neuron.getLayerID(false);
		int neuronLayerIdInHashMap = currentNeuronLayerID + 1;
		HashMap<Integer, int[]> previousConnectedNeuronsHash = network.getNetworkStructure(false).get(neuronLayerIdInHashMap);
		int[] previousConnectedNeurons = previousConnectedNeuronsHash.get(currentNeuronID);
		
		
		if(neuron.getIsInput()){
			
			int currentExample = this.currentExample;
			ArrayList<Double> inputs = network.getExample(currentExample, false);
			for(int i = 0; i< inputs.size(); i++){
				previousLayerOutputs.add(inputs.get(i));
			}
			
		}else{
			
			for(int i = 0; i< previousConnectedNeurons.length; i++){
				
				int currentConnectedNeuron = previousConnectedNeurons[i];
				Perceptron connectedNeuron = network.getSingleNeuron(currentConnectedNeuron, false);
				double connectedNeuronOutput = connectedNeuron.getOutput(false);
				previousLayerOutputs.add(connectedNeuronOutput);
				
			}
			
			double currentNeuronBiasWeight = neuron.getBiasWeight(false);
			previousLayerOutputs.add(currentNeuronBiasWeight);
			
		}
		
		
		if(printInfo){
			System.out.println("PreviousLayer Outs (with 1.0 for bias) is : " + previousLayerOutputs);
		}
		
		return previousLayerOutputs;

	}
	
	public double computeBetaError(Perceptron neuron, boolean printInfo){
		
		
		double betaError = Double.NEGATIVE_INFINITY;
		double derivative = Double.NEGATIVE_INFINITY;
		double out = neuron.getOutput(printBackPropInfo);
		
		if(neuron.getIsOutput()){
		
			ArrayList<Double> outputsY = network.getOutputData(false);
			double expectedY = outputsY.get(currentExample);
			
			derivative = (expectedY - out);
			
			if(printBackPropInfo){
				
				System.out.println("\tDerivative : " + derivative);
			}
			
		}else if(neuron.getIsInput()){
			
			System.out.println(neuron + " is input. Skipping Beta Error");
			
			
		}else{
			
			System.out.println("Computing Beta Error for Normal Neuron : " + neuron);
			
			//// -------   LINES OF CODE TO DECIPHER
			double betaSumsFromNextRow = getBetaSumsFromNextRow(neuron, false);
//			double neuronWeightToNextNeuron = getCurrentNeuronWeightInputFromNextRow(neuron, false);
//			derivative = betaSumsFromNextRow * neuronWeightToNextNeuron; 
//			
//			
//			if(printInfo){
//				
//				System.out.println("(" + out + ") * ( 1 - " + out  + " ) * (" + derivative + ")");
//				System.out.println("... where derivative is " + betaSumsFromNextRow + " + " + neuronWeightToNextNeuron);
//			}
			
		}
		
		betaError = MlUtils.formatDouble((out * ( 1 - out ) * derivative));
		
		if(printInfo){
			System.out.println("\t" + neuron + " beta error set to : " + betaError);
		}
		
		return betaError;
		
	}
	
	
	private double getBetaSumsFromNextRow(Perceptron neuron, boolean printInfo) {
		
		double betaSumsFromNext = 0.0;
		
		network.getNetworkStructure(true);
//		int nextLayerIndex = (neuron.getLayerID(false)) + 1;
//		ArrayList<Perceptron> nextLayer = network.getLayer(nextLayerIndex, false);
//		
//		for(int i = 0; i< nextLayer.size(); i++){
//			
//			Perceptron currentNeuronInNextLayer = nextLayer.get(i);
//			double currentNextBetaVal = currentNeuronInNextLayer.getBetaError(false);
//			betaSumsFromNext += currentNextBetaVal;
//		}
		
		if(printInfo){
			System.out.println(neuron + " Beta Sum From next layer : " + betaSumsFromNext);
		}
		
		return betaSumsFromNext;
	}
	
	
	
	
	
	
	public int getCurrentExample(boolean printInfo){
		
		if(printInfo){
			System.out.println("Current Example Number is : " + this.currentExample);
		}
		return this.currentExample;
		
	}
	
	private void setCurrentExample(int exampleNumber){
		this.currentExample = exampleNumber;
	}

	private int getLastNeuronInNetwork(boolean printInfo){
		
		int lastActualNeuron = 0;
		
		for(int i = 0; i<this.networkSize; i++){
			
			Perceptron currentNeuron = this.network.getNeurons(false).get(i);
			boolean isInput = currentNeuron.getIsInput();
			if(printInfo){
				System.out.println(currentNeuron + " isInput : " + isInput);
			}
			if(!isInput){
				lastActualNeuron = i;
				break;
			}
		}
		
		if(printInfo){
			
			System.out.println("Last actual neuron : " + lastActualNeuron);
		}
		
		return lastActualNeuron;
	}

	private int confirmLastNeuron(int lastActualNeuron, boolean printInfo){
		
	int lastNeuron = lastActualNeuron;
		
		if((test1Neuron) && (this.neuronToTest >= lastNeuron)){
			lastNeuron = this.neuronToTest;
		}
		
		if(printInfo){
			System.out.println("Last neuron set to : " + lastNeuron);
		}
		
		
		return lastNeuron;
	}
}
