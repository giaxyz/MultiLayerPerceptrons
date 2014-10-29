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
		MLNetwork network = new MLNetwork(networkStructure, networkStructureWeights, biasWeights, biasValues, learningRate, momentum);
		
		
	}

}
