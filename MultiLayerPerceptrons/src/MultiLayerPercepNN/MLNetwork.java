package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MLNetwork {
	
	private ArrayList<ArrayList<Double>> inputData;
	private ArrayList<HashMap<Integer,int[]>> networkStructure;
	private ArrayList<HashMap<Integer,double[]>> networkWeights;
	private HashMap<Integer, Double> biasWeights;
	private HashMap<Integer, Double> biasValues;
	private double learningRate;
	private double momentum;
	private ArrayList<Perceptron> allNeurons;
	private ArrayList<Integer> allLayers;
	
	
	public MLNetwork(
			
			ArrayList<ArrayList<Double>> inputData,
			ArrayList<HashMap<Integer,int[]>> networkStructure, 
			ArrayList<HashMap<Integer,double[]>> networkWeights,
			HashMap<Integer, Double> biasWeights,
			HashMap<Integer, Double> biasValues,
			double learningRate,
			double momentum
			) throws Exception{
		
		this.allLayers = new ArrayList<Integer>();
		this.inputData = inputData;
		this.networkStructure = networkStructure;
		this.networkWeights = networkWeights;
		this.biasWeights = biasWeights;
		this.biasValues = biasValues;
		this.learningRate = learningRate;
		this.momentum = momentum;
		allNeurons = new ArrayList<Perceptron>();
	
		createNetwork(true); // automatically populates allNeurons
		
	}
	
	
	
	public void feedForward(int exampleIndex, boolean printInfo) {
		
		if(printInfo){
			
			System.out.println("\n\t\t-----FeedingForward Example Index : " + exampleIndex);
		}
		
			for(int i = -1; i<(getLayerIndices(false).size() -1); i++){
				
				int layerIndex = i;
				
				if(printInfo){
					System.out.print("\n");
					System.out.print("\t ---- at layer : " + layerIndex);
					if(layerIndex == -1){
						System.out.print("  // where -1 indicates Input Layer");
					}
					System.out.print("\n\n\n");
				}
				
				feedForwardLayer(layerIndex, exampleIndex, false);
				
			}
		}
		
	


	private void feedForwardLayer(int layerIndex, int exampleIndex, boolean printInfo) {
	
		if(printInfo){
			System.out.println("\t\t -- Current Example : " + exampleIndex);
		}
		
		ArrayList<Double> currentInputsX = getExample(exampleIndex, false);
		
		//setLayerNeuronInputs(layerIndex, exampleIndex);
		//computeSumsinLayer(layerIndex);
	}
	
	public void createNetwork(boolean printInfo) throws Exception {
		
		for(int i = 0; i< this.networkStructure.size(); i++){
			
			HashMap<Integer,int[]> currentLayer= networkStructure.get(i);
			
			
			Iterator<Map.Entry<Integer, int[]>> iterator = currentLayer.entrySet().iterator() ;
			
			while(iterator.hasNext()){
				
				Map.Entry<Integer, int[]> mapEnt = iterator.next();
				int neuronID = mapEnt.getKey();
				double neuronBiasWeight = biasWeights.get(neuronID);
				double neuronBiasValue = biasValues.get(neuronID);
				int layerID = getNeuronLayer(neuronID, false);
				boolean isInput = getIsInput(neuronID, false);
				boolean isOutput = getIsOutput(neuronID, layerID, false);
				int[] neuronConnections = mapEnt.getValue();
				double[] neuronWeightsMap = getNeuronWeights(neuronID, layerID, false);
				
				//System.out.print("\nNeuron" + neuronID + " Connections are : " );
				//MlUtils.printIntArray(neuronConnections);
				
				createNeuron(neuronID,
						layerID,
						isInput,
						isOutput,
						neuronConnections,
						neuronWeightsMap,
						neuronBiasWeight, 
						neuronBiasValue,
						this.inputData);
				
			}
			
			int layerIndex = (i - 1);
			allLayers.add(layerIndex);
			
		}
      
	}
	
	private double[] getNeuronWeights(int neuronID, int layerID, boolean printInfo) {
			
		double[] weights = null;
			
			for(int i = 0; i<networkWeights.size(); i++){
				
				HashMap<Integer,double[]> currentMap = networkWeights.get(i);
				if(currentMap.containsKey(neuronID)){
					weights = currentMap.get(neuronID);
					
					
				}
				
			}
			
			if(printInfo){
				System.out.println("\nNeuronID " + neuronID + ", Layer " + layerID +  " Weights :");
				MlUtils.printDoubleArray(weights);
			}
			
			return weights;
		}

	private boolean getIsOutput(int neuronID, int layerID, boolean printInfo){
		
			boolean isOutput = false;
			int lastLayer = networkStructure.size() - 2;
		
			if(layerID == lastLayer){
			isOutput = true;
			if(printInfo){
				
				System.out.println("\nLast Layer Neuron : " + neuronID + " Layer : " + layerID);
			}
			
		}
		
		
		return isOutput;
	}
	
	private boolean getIsInput(int neuronID, boolean printInfo){
		
		boolean isInput = false;
		HashMap<Integer,int[]> currentNeuronWeightInfo = networkStructure.get(0);
			
		if(currentNeuronWeightInfo.containsKey(neuronID)){
			
			isInput = true;
		}
		
		if(printInfo){
			System.out.println("NeuronID " + neuronID + " isInput = " + isInput);
		}
				
	
		return isInput;
	}
	
	private int getNeuronLayer(int neuronID, boolean printInfo) throws Exception {
		
		int layerID = -5;
		
		for(int i = 0; i<this.networkStructure.size(); i++){
			
			HashMap<Integer,int[]> currentNeuronWeightInfo = networkStructure.get(i);
			
			if(currentNeuronWeightInfo.containsKey(neuronID)){
				
				layerID =  (i-1);
			}
				
		}
		
		if(layerID == -5){
			throw new Exception("Error, Neuron " + neuronID + "Layer info not found in Hash Map");
		}
		
		if(printInfo){
			System.out.println("NeuronID" + neuronID + " Layer at " + layerID);
		}
		
		return layerID;
		
	}

	public Perceptron createNeuron(
			int neuronID,
			int layerID,
			boolean isInput,
			boolean isOutput,
			int[] neuronConnections,
			double[] neuronWeights,
			double biasValue, 
			double biasWeight,
			ArrayList<ArrayList<Double>> inputData){
		
		Perceptron neuron = new Perceptron(
				biasWeight, biasValue,
				neuronID, layerID,  
				isOutput, isInput,
				neuronConnections, neuronWeights, 
				inputData);
		
		this.allNeurons.add(neuron);
		return neuron;
	}
	

	// ----- All Get Methods ---------------------------------------------------------
	
	public ArrayList<Perceptron> getNeurons(boolean printInfo){
		
		if(printInfo){
			for(int i = 0; i< this.allNeurons.size(); i++){
				System.out.println(allNeurons.get(i));
			}
		}
		
		return this.allNeurons;
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
				
				HashMap<Integer,int[]> currentNeuronConnectionsInfo = networkStructure.get(i);
				System.out.println("\nNetwork Connections Layer " + (i-1) + " : ");
				MlUtils.printHashMapIntIntArray(currentNeuronConnectionsInfo);;
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

	public ArrayList<Integer> getLayerIndices(boolean printInfo){
		if(printInfo){
			System.out.println("Network Layer Indices, where -1 indicates input layer : " + this.allLayers );
		}
		return this.allLayers;
	}
	
	public ArrayList<ArrayList<Double>> getInputData(boolean printInfo){
		
		if(printInfo){
			System.out.println("NMaker Input data is :" + this.inputData);
		}
		return this.inputData;
	}
	



	public ArrayList<Double> getExample(int exampleIndex, boolean printInfo){
		
		ArrayList<Double> currentRowInputs = new ArrayList<Double>();
		for(int i = 0; i< (this.inputData.size()-1); i++){ // we minus 1 here because the last one is the output Y column
			
			ArrayList<Double> currentColumnInputs = this.inputData.get(i);
			currentRowInputs.add(currentColumnInputs.get(exampleIndex));
			
			if(printInfo){
				System.out.println("Data at column : " + i + " "+ currentColumnInputs);
			}
			
		}
		
		if(printInfo){
			System.out.println("   extracted data at row : " + exampleIndex + " " + currentRowInputs);
			System.out.println("\n");
		}
		
		return currentRowInputs;
	}



	public int getNumberOfExamples(boolean printInfo) {
		
		ArrayList<Double> firstInputData = getInputData(false).get(0);
		int numberOfExamples = firstInputData.size();
		
		if(printInfo){
			System.out.println("Number of egs in input Data :  " + numberOfExamples );
		}
		
		return numberOfExamples;
		
	}
}
