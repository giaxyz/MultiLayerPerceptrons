package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class NetworkMaker {
	
	private ArrayList<ArrayList<Double>> data;
	private double learningRate;
	private double momentum;
	ArrayList<HashMap<Integer,int[]>> networkStructure;
	ArrayList<HashMap<Integer,double[]>> networkStructureWeights;
	private HashMap<Integer, Double> biasWeights;
	private HashMap<Integer, Double> biasValues;
	private ArrayList<Double> outputs;
	
	public NetworkMaker(String networkName) throws Exception{
		
		
		this.data = new ArrayList<ArrayList<Double>>();
		this.networkStructure = new ArrayList<HashMap<Integer,int[]>>();
		this.networkStructureWeights = new ArrayList<HashMap<Integer,double[]>>();
		this.biasWeights = new HashMap<Integer, Double>();
		this.biasValues = new HashMap<Integer, Double>();
		
		this.learningRate = 1.0;
		this.momentum = 0.0;
		
		if(networkName.equals("SelectedConnections")){
			
			//System.out.println("--------Setting up Gia's 3 x 1 Network Configuration--------");
			this.data = selectedConnectionsInputData();
			this.networkStructure = selectedConnectionsNetworkStructure();
			this.networkStructureWeights = selectedConnectionNetworkStructureWeights();
			this.biasValues = selectedConnectionsBiasValues();
			this.biasWeights = selectedConnectionsBiasWeights();
			this.outputs = selectedConnectionsOutputs();
		
		}else if (networkName.equals("FullyConnected")){
			
			//System.out.println("--------Setting up Gia's 2 x 1 FullyConnected Configuration------");
			this.data = fullyConnectedInputData();
			this.networkStructure = fullyConnectedNetworkStructure();
			this.networkStructureWeights = fullyConnectedNetworkStructureWeights();
			this.biasValues = fullyConnectedBiasValues();
			this.biasWeights = fullyConnectedBiasWeights();
			this.outputs = fullyConnectedOutputs();
			
		}else{
			
			throw new Exception("Error, You have yet to design this network.  See NetworkMaker Class");
		}
	}
	
	
	// ------ Set Methods for the "Fully Connected" Network, so named, all hard coded
	// This section needs duplicating and setting for any other network, and you 
	// have to know what you're doing in order to set this up properly
	private ArrayList<Double> fullyConnectedOutputs(){
		ArrayList<Double> outputs = new ArrayList<Double>();
		outputs.add(0.0);
		outputs.add(1.0);
		return outputs;
	}
	
	private HashMap<Integer, Double> fullyConnectedBiasWeights(){

		HashMap<Integer, Double> biasWeights = new HashMap<Integer, Double>();
		biasWeights.put(0,1.0);
		biasWeights.put(1,1.0);
		biasWeights.put(2,-0.1);
		biasWeights.put(3,-0.15);
		biasWeights.put(4,0.05);
		return biasWeights;
	}
	
	private HashMap<Integer, Double> fullyConnectedBiasValues(){
		
		HashMap<Integer, Double> biasValues = new HashMap<Integer, Double>();
		biasValues.put(0,1.0);
		biasValues.put(1,1.0);
		biasValues.put(2,1.0);
		biasValues.put(3,1.0);
		biasValues.put(4,1.0);
		return biasValues;
	}
	
	private ArrayList<HashMap<Integer,double[]>> fullyConnectedNetworkStructureWeights(){
		
		ArrayList<HashMap<Integer,double[]>> networkStructWeightsHere = new ArrayList<HashMap<Integer,double[]>>();
		
		// Input Layer Weights Map
		HashMap<Integer,double[]> InputLayerWeights = new HashMap<Integer, double[]>();
		InputLayerWeights.put(0, null); // put the neuron ID, and then the weights of the respective IDs it is connected to
		InputLayerWeights.put(1, null);
		
		// Layer 0 Weights Map
		HashMap<Integer,double[]> Layer0Weights = new HashMap<Integer, double[]>();
		double[] connectionWeights2 = {-0.1,0.2};
		double[] connectionWeights3 = {0.3,-0.1};
		Layer0Weights.put(2, connectionWeights2);
		Layer0Weights.put(3, connectionWeights3);
		
		
		// Output Layer Weights Map
		HashMap<Integer,double[]> OutputLayerWeights = new HashMap<Integer, double[]>();
		double[] outputWeights = {-0.1,0.2};
		OutputLayerWeights.put(4, outputWeights);
		
		// Add all layers to the network structure
		networkStructWeightsHere.add(InputLayerWeights);
		networkStructWeightsHere.add(Layer0Weights);
		networkStructWeightsHere.add(OutputLayerWeights);
		
		return networkStructWeightsHere;
	}

	private ArrayList<HashMap<Integer,int[]>> fullyConnectedNetworkStructure(){
		
		ArrayList<HashMap<Integer,int[]>> networkStructureSelected = new ArrayList<HashMap<Integer,int[]>>();
		
		// -- Input Layer Connections
		
		HashMap<Integer,int[]> InputLayer = new HashMap<Integer, int[]>();
		InputLayer.put(0, null); // put the neuron ID, and then the IDs of the ones it is connected to
		InputLayer.put(1, null);
		
		// -- Layer 0 Connections
		
		HashMap<Integer,int[]> Layer0 = new HashMap<Integer, int[]>();
		int[] connections2 = {0,1};
		int[] connections3 = {0,1};
		Layer0.put(2, connections2);
		Layer0.put(3, connections3);
	
		
		// -- Output Layer Connections
		
		HashMap<Integer,int[]> OutputLayer = new HashMap<Integer, int[]>();
		int[] connections4 = {2,3};
		OutputLayer.put(4,connections4);
		
		networkStructureSelected.add(InputLayer);
		networkStructureSelected.add(Layer0);
		networkStructureSelected.add(OutputLayer);
		
		return networkStructureSelected;
	}
	
	private ArrayList<ArrayList<Double>> fullyConnectedInputData() throws Exception{
		
		ArrayList<ArrayList<Double>> inputData = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> inputsX1 =  new ArrayList<Double>(Arrays.asList(0.0,0.0));
		ArrayList<Double> inputsX2 =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		ArrayList<Double> outputs =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		inputData.add(inputsX1);
		inputData.add(inputsX2);
		inputData.add(outputs);
		InputDataChecker inputDataCheck = new InputDataChecker(inputData); // put the input data checker in the utilities class
		inputDataCheck.checkData();
		inputDataCheck.getData(false);
		
		return inputData;
		
	}
	
	
	// ------ Set Methods for the "Selected Connections" Network, so named, all hard coded
	// This section needs duplicating and setting for any other network, and you 
	// have to know what you're doing in order to set this up properly
	
	private ArrayList<Double> selectedConnectionsOutputs(){
		
		ArrayList<Double> outputs = new ArrayList<Double>();
		outputs.add(0.0);
		outputs.add(1.0);
		return outputs;
	}
	
	private HashMap<Integer, Double> selectedConnectionsBiasWeights(){

		HashMap<Integer, Double> biasWeights = new HashMap<Integer, Double>();
		biasWeights.put(0,0.0);
		biasWeights.put(1,0.0);
		biasWeights.put(2,0.0);
		biasWeights.put(3,0.0);
		biasWeights.put(4,0.0);
		biasWeights.put(5,0.0);
		return biasWeights;
	}
	
	private HashMap<Integer, Double> selectedConnectionsBiasValues(){
		
		HashMap<Integer, Double> biasValues = new HashMap<Integer, Double>();
		biasValues.put(0,1.0);
		biasValues.put(1,1.0);
		biasValues.put(2,1.0);
		biasValues.put(3,1.0);
		biasValues.put(4,1.0);
		biasValues.put(5,1.0);
		return biasValues;
	}
	
	private ArrayList<HashMap<Integer,double[]>> selectedConnectionNetworkStructureWeights(){
		
		ArrayList<HashMap<Integer,double[]>> networkStructWeightsHere = new ArrayList<HashMap<Integer,double[]>>();
		
		// Input Layer Weights Map
		HashMap<Integer,double[]> InputLayerWeights = new HashMap<Integer, double[]>();
		double[] connectionWeights0 = null;
		double[] connectionWeights1 = null;
		InputLayerWeights.put(0, connectionWeights0); // put the neuron ID, and then the weights of the respective IDs it is connected to
		InputLayerWeights.put(1, connectionWeights1);
		
		// Layer 0 Weights Map
		HashMap<Integer,double[]> Layer0Weights = new HashMap<Integer, double[]>();
		double[] connectionWeights2 = {0.2};
		double[] connectionWeights3 = {0.2};
		double[] connectionWeights4 = {0.1,0.2};
		Layer0Weights.put(2, connectionWeights2);
		Layer0Weights.put(3, connectionWeights3);
		Layer0Weights.put(4, connectionWeights4);
		
		// Output Layer Weights Map
		HashMap<Integer,double[]> OutputLayerWeights = new HashMap<Integer, double[]>();
		double[] outputWeights = {0.2,0.1,-0.2,0.2};
		OutputLayerWeights.put(5, outputWeights);
		
		
		networkStructWeightsHere.add(InputLayerWeights);
		networkStructWeightsHere.add(Layer0Weights);
		networkStructWeightsHere.add(OutputLayerWeights);
		
		return networkStructWeightsHere;
	}

	private ArrayList<HashMap<Integer,int[]>> selectedConnectionsNetworkStructure(){
		
		ArrayList<HashMap<Integer,int[]>> networkStructureSelected = new ArrayList<HashMap<Integer,int[]>>();
		
		// -- Input Layer Connections
		
		HashMap<Integer,int[]> InputLayer = new HashMap<Integer, int[]>();
		int[] connections0 = null;
		int[] connections1 = null;
		InputLayer.put(0, connections0); // put the neuron ID, and then the IDs of the ones it is connected to
		InputLayer.put(1, connections1);
		
		// -- Layer 0 Connections
		
		HashMap<Integer,int[]> Layer0 = new HashMap<Integer, int[]>();
		int[] connections2 = {0};
		int[] connections3 = {1};
		int[] connections4 = {0,1};
		Layer0.put(2, connections2);
		Layer0.put(3, connections3);
		Layer0.put(4, connections4);
		
		// -- Output Layer Connections
		
		HashMap<Integer,int[]> OutputLayer = new HashMap<Integer, int[]>();
		int[] connections5 = {0,2,3,4};
		OutputLayer.put(5,connections5);
		
		networkStructureSelected.add(InputLayer);
		networkStructureSelected.add(Layer0);
		networkStructureSelected.add(OutputLayer);
		
		return networkStructureSelected;
	}
	
	private ArrayList<ArrayList<Double>> selectedConnectionsInputData() throws Exception{
		
		ArrayList<ArrayList<Double>> inputData = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> inputsX1 =  new ArrayList<Double>(Arrays.asList(1.0,1.0));
		ArrayList<Double> inputsX2 =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		ArrayList<Double> outputs =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		inputData.add(inputsX1);
		inputData.add(inputsX2);
		inputData.add(outputs);
		InputDataChecker inputDataCheck = new InputDataChecker(inputData); // put the input data checker in the utilities class
		inputDataCheck.checkData();
		inputDataCheck.getData(false);
		
		return inputData;
		
	}

	/// --- All Get Methods ---------------------------------------------------------------
	
	public ArrayList<ArrayList<Double>> getInputData(boolean printInfo){
		
		if(printInfo){
			System.out.println("NMaker Input data is :" + this.data);
		}
		return this.data;
	}
	
	public ArrayList<HashMap<Integer,int[]>> getSelectedConnectionsNetworkStructure(boolean printInfo){
		
		if(printInfo){
			System.out.println("Network Structure : " + this.networkStructure);
		} 
		return this.networkStructure;
		
	}

	public ArrayList<HashMap<Integer,double[]>> getSelectedConnectionsNetworkStructureWeights(boolean printInfo){
		
		if(printInfo){
			
			System.out.println("Network Structure Weigths : " + this.networkStructureWeights);
		} 
		return this.networkStructureWeights;
	}
	
	public double getLearningRate(boolean printInfo){
		if(printInfo){
			System.out.println("Learning Rate : " + this.learningRate);
		}
		return this.learningRate;
	}

	public double getMomentum(boolean printInfo){
		if(printInfo){
			System.out.println("Momentum : " + momentum);
		}
		return this.momentum;
	}

	public HashMap<Integer, Double> getBiasValues(boolean printInfo){
		
		if(printInfo){
			System.out.println("Bias Values : " + this.biasValues);
		}
		return this.biasValues;
	}

	public HashMap<Integer, Double> getBiasWeights(boolean printInfo){
		
		if(printInfo){
			System.out.println("Bias Weights : " + this.biasWeights);
		}
		return this.biasWeights;
	}

	public ArrayList<Double> getOutputs(boolean printInfo){
		
		if(printInfo){
			System.out.println("Data Outputs ordered by Example Are : " + this.outputs);
		}
		return this.outputs;
	}
}





