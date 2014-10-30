package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.HashMap;

public class MLPMain {

	public static void main(String[] args) throws Exception {
		
		//   ------------ Set up to Read all Data ------------------------------------------
		
		//NetworkMaker networkMaker = new NetworkMaker("SelectedConnections");
		NetworkMaker networkMaker = new NetworkMaker("FullyConnected");
		ArrayList<ArrayList<Double>> data = networkMaker.getInputData(false);
		ArrayList<HashMap<Integer,int[]>> networkStructure = networkMaker.getSelectedConnectionsNetworkStructure(false);
		ArrayList<HashMap<Integer,double[]>> networkStructureWeights = networkMaker.getSelectedConnectionsNetworkStructureWeights(false);
		double learningRate = networkMaker.getLearningRate(false);
		double momentum = networkMaker.getMomentum(false);
		HashMap<Integer, Double> biasWeights = networkMaker.getBiasWeights(false);
		HashMap<Integer, Double> biasValues = networkMaker.getBiasValues(false);
		
		// Create the network
		boolean printFeedForward = true;
		MLNetwork network = new MLNetwork(data, networkStructure, networkStructureWeights, biasWeights, biasValues, learningRate, momentum, printFeedForward);
		//network.getNeurons(false);
		int numberOfExamples = network.getNumberOfExamples(false);
		numberOfExamples = 1; // Overwriting the number of examples here
		boolean test1Layer = false; // if on, will only test 1 layer
		boolean printInfo = false;
		
		int layerToTest = 0; // set the layer to test.  -1 is the input layer
		int numberOfEpochs = 1;
		
		for(int i = 0; i< numberOfEpochs; i++){
			
			System.out.println(" \n\n\n -------Begin Training, epoch number : " + i + " -------\n\n ");
			
			for(int j = 0; j<numberOfExamples; j++ ){
				
				network.feedForward(j, printInfo, test1Layer, layerToTest);
				
				// -- To DO -----------
				//System.out.println("\t\t Backpropagating");
				//System.out.println("\t\t\t\t Beta Errors : ");
			}
			
			
			
		}
		
		
	}

}
