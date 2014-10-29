package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class MLPMain {

	public static void main(String[] args) throws Exception {
		
		
		
		//   ------------ Set up to Read all Data ------------------------------------------
		
		NetworkMaker networkMaker = new NetworkMaker("SelectedConnections");
		ArrayList<ArrayList<Double>> data = networkMaker.getInputData(true);
		ArrayList<HashMap<Integer,int[]>> networkStructure = networkMaker.getSelectedConnectionsNetworkStructure(true);
		ArrayList<HashMap<Integer,double[]>> networkStructureWeights = networkMaker.getSelectedConnectionsNetworkStructureWeights(true);
		double learningRate = networkMaker.getLearningRate(true);
		double momentum = networkMaker.getMomentum(true);
		HashMap<Integer, Double> biasWeights = networkMaker.getBiasWeights(true);
		HashMap<Integer, Double> biasValues = networkMaker.getBiasValues(true);
		

		
		
		MLNetwork network = new MLNetwork(networkStructure, networkStructureWeights, biasWeights, biasValues, learningRate, momentum);
		
		
	}

}
