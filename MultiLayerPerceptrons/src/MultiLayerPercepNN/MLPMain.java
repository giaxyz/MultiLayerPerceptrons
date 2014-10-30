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
		MLNetwork network = new MLNetwork(data, networkStructure, networkStructureWeights, biasWeights, biasValues, learningRate, momentum);
		network.getNeurons(true);
		int numberOfExamples = network.getNumberOfExamples(false);
		numberOfExamples = 1; // Overwriting the number of examples here
		int numberOfEpochs = 1;
		
		for(int i = 0; i< numberOfEpochs; i++){
			
			//System.out.println(" \n ---- Training, epoch number : " + i + " ------- ");
			
			for(int j = 0; j<numberOfExamples; j++ ){
				
				network.feedForward(j, false);
				
				// -- To DO -----------
				//System.out.println("\t\t Backpropagating");
				//System.out.println("\t\t\t\t Beta Errors : ");
			}
			
			
			
		}
		
		
	}

}
