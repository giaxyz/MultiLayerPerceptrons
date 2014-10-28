package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class MLPMain {

	public static void main(String[] args) throws Exception {
		
		//   ------------ Set up to Read all Data ------------------------------------------
		
		ArrayList<ArrayList<Double>> data = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> inputsX1 =  new ArrayList<Double>(Arrays.asList(1.0,1.0));
		ArrayList<Double> inputsX2 =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		ArrayList<Double> outputs =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		data.add(inputsX1);
		data.add(inputsX2);
		data.add(outputs);
		InputDataChecker inputData = new InputDataChecker(data); // put the input data checker in the utilities class
		inputData.checkData();
		inputData.getData(true);
		
		//     ---------------   Set up the Layer connections and Weights  -----------------------------
		
		// -- Input Layer
		
		HashMap<Integer,int[]> InputLayer = new HashMap<Integer, int[]>();
		int[] connections0 = null;
		int[] connections1 = null;
		InputLayer.put(0, connections0); // put the neuron ID, and then the IDs of the ones it is connected to
		InputLayer.put(1, connections1);
		
		HashMap<Integer,double[]> InputLayerWeights = new HashMap<Integer, double[]>();
		double[] connectionWeights0 = null;
		double[] connectionWeights1 = null;
		InputLayerWeights.put(0, connectionWeights0); // put the neuron ID, and then the weights of the respective IDs it is connected to
		InputLayerWeights.put(1, connectionWeights1);
		
		// -- Layer 0
		
		HashMap<Integer,int[]> Layer0 = new HashMap<Integer, int[]>();
		int[] connections2 = {0};
		int[] connections3 = {1};
		int[] connections4 = {0,1};
		Layer0.put(2, connections2);
		Layer0.put(3, connections3);
		Layer0.put(4, connections4);
		
		HashMap<Integer,double[]> Layer0Weights = new HashMap<Integer, double[]>();
		double[] connectionWeights2 = {0.2};
		double[] connectionWeights3 = {0.2};
		double[] connectionWeights4 = {0.1,0.2};
		Layer0Weights.put(2, connectionWeights2);
		Layer0Weights.put(3, connectionWeights3);
		Layer0Weights.put(4, connectionWeights4);
		
		// -- Output Layer
		
		HashMap<Integer,int[]> OutputLayer = new HashMap<Integer, int[]>();
		int[] connections5 = {0,2,3,4};
		OutputLayer.put(5,connections5);
		
		HashMap<Integer,double[]> OutputLayerWeights = new HashMap<Integer, double[]>();
		double[] outputWeights = {0.2,0.1,-0.2,0.2};
		OutputLayerWeights.put(5, outputWeights);
		
		ArrayList<HashMap<Integer,int[]>> networkStructure = new ArrayList<HashMap<Integer,int[]>>();
		networkStructure.add(InputLayer);
		networkStructure.add(Layer0);
		networkStructure.add(OutputLayer);
		
		ArrayList<HashMap<Integer,double[]>> networkStructWeights = new ArrayList<HashMap<Integer,double[]>>();
		networkStructWeights.add(InputLayerWeights);
		networkStructWeights.add(Layer0Weights);
		networkStructWeights.add(OutputLayerWeights);
		
		// Set up the Bias Weights
		
		HashMap<Integer, Double> biasWeights = new HashMap<Integer, Double>();
		biasWeights.put(0,0.0);
		biasWeights.put(1,0.0);
		biasWeights.put(2,0.0);
		biasWeights.put(3,0.0);
		biasWeights.put(4,0.0);
		biasWeights.put(5,0.0);
		
		// Set up the Bias Values
		
		HashMap<Integer, Double> biasValues = new HashMap<Integer, Double>();
		biasWeights.put(0,1.0);
		biasWeights.put(1,1.0);
		biasWeights.put(2,1.0);
		biasWeights.put(3,1.0);
		biasWeights.put(4,1.0);
		biasWeights.put(5,1.0);
		
		// ---   Last of the variables
		
		double learningRate = 1.0;
		double momentum = 0.0;
		
		
		// setup this whole area so it's a separate class.  That way you can check his online version of 
		// the other network
		// --> so, assignment data class. and also test data class.  
		
		MLNetwork network = new MLNetwork(networkStructure, networkStructWeights, biasWeights, biasValues, learningRate, momentum);
		
		
	}

}
