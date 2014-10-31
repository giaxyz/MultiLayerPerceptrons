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
	private boolean printFeedForwardDetails;
	private ArrayList<Double> outputs;
	
	
	public MLNetwork(
			
			ArrayList<ArrayList<Double>> inputData,
			ArrayList<Double> outputs,
			ArrayList<HashMap<Integer,int[]>> networkStructure, 
			ArrayList<HashMap<Integer,double[]>> networkWeights,
			HashMap<Integer, Double> biasWeights,
			HashMap<Integer, Double> biasValues,
			double learningRate,
			double momentum,
			boolean printFeedForward
			) throws Exception{
		
		this.printFeedForwardDetails = printFeedForward;
		this.allLayers = new ArrayList<Integer>();
		this.inputData = inputData;
		this.outputs = outputs;
		this.networkStructure = networkStructure;
		this.networkWeights = networkWeights;
		this.biasWeights = biasWeights;
		this.biasValues = biasValues;
		this.learningRate = learningRate;
		this.momentum = momentum;
		allNeurons = new ArrayList<Perceptron>();
	
		createNetwork(true); // automatically populates allNeurons
		
	}
	
	
	
	public void feedForward(int exampleIndex, boolean printInfo, boolean test1Layer, int layerToTest) throws Exception {
		
		if(printInfo){
			
			System.out.println("\n\t\t-----FeedingForward Example Index : " + exampleIndex);
		}
		
		int maxLayer = -1;
		
		if(test1Layer){
			maxLayer = layerToTest + 1;
		}else{
			maxLayer = getLayerIndices(false).size() - 1;
		}
			
			for(int i = -1; i<maxLayer; i++){
				
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
		
	private void feedForwardLayer(int layerIndex, int exampleIndex, boolean printInfo) throws Exception {
	
		if(this.printFeedForwardDetails){
			
			System.out.println("---------------------------------- Feeding forward layer : " + layerIndex);
			//System.out.println("\t\t -- Current Example : " + exampleIndex);
		}
		
		ArrayList<Double> currentInputsX = getExample(exampleIndex, false);
		setLayerNeuronInputs(currentInputsX, layerIndex, exampleIndex, this.printFeedForwardDetails);
		computeSumsinLayer(layerIndex, this.printFeedForwardDetails);
	}
	
	private void computeSumsinLayer(int layerIndex, boolean printInfo) throws Exception {
		
		
		
		ArrayList<Perceptron> layerNeurons = getLayerNeurons(layerIndex, false);
		for(int i = 0; i< layerNeurons.size(); i++){
			
			Perceptron neuron = layerNeurons.get(i);
			double sum = neuron.computeSum(this.printFeedForwardDetails);
			neuron.activateSigmoid(sum, this.printFeedForwardDetails);
			neuron.getOutput(this.printFeedForwardDetails);
			if(this.printFeedForwardDetails){
				System.out.println("\n\t\t-------");
			}
		}
			

	

		
	}

	private void setLayerNeuronInputs(ArrayList<Double> currentInputsX, int layerIndex, int exampleIndex, boolean printInfo){
		
		if(printInfo){
			
			//System.out.println("Setting inputs for Layer : " + layerIndex);
			
		}
			ArrayList<Perceptron> currentLayer = getLayerNeurons(layerIndex, false);
			
			
			if(layerIndex == -1){ // If it is an input layer, it's a dummy neuron.  Set to inputs
				
				ArrayList<Double> inputLayerInputs = MlUtils.duplicateArrayList(currentInputsX);
				
				//System.out.println("\tRaw Inputs:" + inputLayerInputs);
				
				for(int j = 0; j< currentLayer.size(); j++){
					
					Perceptron neuron = currentLayer.get(j);
					//System.out.println(" Input layer inputs " + inputLayerInputs);
					double inputValue = inputLayerInputs.get(j);
					//System.out.println("   at value : " + inputValue);
				
					// Since it's an Input Dummy Neuron,
					// Set the output value to the Raw input X value
				
					neuron.setOutput(inputValue);
					neuron.setSum(inputValue);
					
					if(this.printFeedForwardDetails){
						System.out.print("Inputs set to : ");
						neuron.getOutput(printFeedForwardDetails);
					}
					
					neuron.getSum(false);
					
				}
				
			}else{ ////!!!!!!!!!1 Otherwise, find out from the hash map, what the connections are
				
				
				
				for(int i = 0; i< currentLayer.size(); i++){
					
					ArrayList<Double> inputsForCurrentNeuron = new ArrayList<Double>();
					Perceptron neuron = currentLayer.get(i);
					if(this.printFeedForwardDetails){
						System.out.println("\nCurrent Neuron: " + neuron);
					}
					
					int layerHashMapIndex = layerIndex + 1; // because the first array is the inputs at -1
					//System.out.print("    -- at layer : " + layerHashMapIndex + "\n");
					int[] neuronConnectedNeurons = getNetworkStructure(false).get(layerHashMapIndex).get(neuron.getNeuronID(false));
					
					for(int j = 0; j< neuronConnectedNeurons.length; j++){
						
						int connectedNeuronIndex = neuronConnectedNeurons[j];
						Perceptron connectedNeuron = getNeuronID(connectedNeuronIndex, false);
						//System.out.println("\tConnection : " + connectedNeuron);
						double outputOfConnectedNeuron = connectedNeuron.getOutput(false);
						inputsForCurrentNeuron.add(outputOfConnectedNeuron);
						
					}
					
					double currentBiasValue = neuron.getBiasValue(false);
					inputsForCurrentNeuron.add(currentBiasValue);
					//System.out.println("\t\t "+ inputsForCurrentNeuron);
					neuron.setInputs(inputsForCurrentNeuron);
					if(this.printFeedForwardDetails){
						System.out.print("\t\t");
					}
					neuron.getInputs(this.printFeedForwardDetails);
				}
				
			}
		
	}
	
	private Perceptron getNeuronID(int connectedNeuronIndex, boolean printInfo) {
		
		
		
		for(int i = 0; i<allNeurons.size(); i++){
			
			Perceptron currentNeuron = allNeurons.get(i);
			int ID = currentNeuron.getNeuronID(false);
			if(ID == connectedNeuronIndex){
				
				if(printInfo){
					System.out.println(currentNeuron);
				}
				
				return currentNeuron;
			}
		}
		
		return null;
	}

	public ArrayList<Perceptron> getLayerNeurons(int layerIndex, boolean printInfo){
		
		ArrayList<Perceptron> currentLayer = new ArrayList<Perceptron>();

		for(int i = 0; i<this.allNeurons.size(); i++){
			
			Perceptron currentPerceptron = allNeurons.get(i);
			int currentPerceptronLayer = currentPerceptron.getLayerID(false);
			
			if(currentPerceptronLayer == layerIndex){
				currentLayer.add(currentPerceptron);
			}
			
		}
		
		if(printInfo){
			System.out.println("Current Layer Neurons in Layer : " + layerIndex + " : ");
			System.out.println("\t\t" + currentLayer);
		}
		
		return currentLayer;
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
				double[] neuronWeightsMapWithoutBias = getNeuronWeights(neuronID, layerID, false);
				double[] neuronWeightsMap = MlUtils.addDoubleToArray(neuronWeightsMapWithoutBias, neuronBiasValue);
			
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
	
	public Perceptron getSingleNeuron(int neuronID, boolean printInfo){
		
		
		for(int i = 0; i< this.allNeurons.size(); i++){
			
			Perceptron currentPerceptron = allNeurons.get(i);
			
			if(currentPerceptron.getNeuronID(false) == neuronID){
				
				if(printInfo){
					
					System.out.println("\t\t---Getting single neuron " + currentPerceptron.getNeuronDetails(false) );
				}
				
				return currentPerceptron;
			}
		}
		
		return null;
		
	}
	
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

	public ArrayList<Double> getOutputData(boolean printInfo){
		
		if(printInfo){
			System.out.println("Outputs are : " + this.outputs);
		}
		
		return this.outputs;
		
	}

	public double getLearningRate(boolean printInfo){
		
		if(printInfo){
			System.out.println("Learning rate is : " + this.learningRate);
		}
		return this.learningRate;
	}




}
