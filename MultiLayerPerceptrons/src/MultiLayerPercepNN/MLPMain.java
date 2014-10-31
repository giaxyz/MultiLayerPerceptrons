package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.HashMap;

public class MLPMain {

	public static void main(String[] args) throws Exception {
		
		//   ------------ Set up to Read all Data ------------------------------------------
		
		//NetworkMaker networkMaker = new NetworkMaker("SelectedConnections");
		NetworkMaker networkMaker = new NetworkMaker("FullyConnected");
		ArrayList<ArrayList<Double>> data = networkMaker.getInputData(false);
		ArrayList<Double> outputs = networkMaker.getOutputs(true);
		ArrayList<HashMap<Integer,int[]>> networkStructure = networkMaker.getSelectedConnectionsNetworkStructure(false);
		ArrayList<HashMap<Integer,double[]>> networkStructureWeights = networkMaker.getSelectedConnectionsNetworkStructureWeights(false);
		double learningRate = networkMaker.getLearningRate(false);
		double momentum = networkMaker.getMomentum(false);
		HashMap<Integer, Double> biasWeights = networkMaker.getBiasWeights(false);
		HashMap<Integer, Double> biasValues = networkMaker.getBiasValues(false);
		
		
		// Settings for Feed Forward
		boolean printFeedForward = false;
		int numberOfEpochs = 1;
		boolean test1Layer = false; // if on, will only test 1 layer
		int layerToTest = 0; // set the layer to test.  -1 is the input layer
		boolean printInfo = false;
		MLNetwork network = new MLNetwork(data, outputs, networkStructure, 
				networkStructureWeights, biasWeights, biasValues, 
				learningRate, momentum, printFeedForward);
		int numberOfExamples = network.getNumberOfExamples(false);
		numberOfExamples = 1; // Overwriting the number of examples here
		
		
		
		for(int i = 0; i< numberOfEpochs; i++){
			
			System.out.println(" \n\n\n -------Begin Training, epoch number : " + i + " -------\n\n ");
			
			for(int j = 0; j<numberOfExamples; j++ ){
				
				int exampleNumber = j;
				network.feedForward(exampleNumber, printInfo, test1Layer, layerToTest);
				Backpropagator backpropagate = new Backpropagator(network);
				backpropagate.backpropagate(exampleNumber);
			
			}
			
		}
		
		
	}

}
