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
		
		//     ---------------   Set up the Layer connections -----------------------------
		
		
		HashMap<Integer,int[]> InputLayer = new HashMap<Integer, int[]>();
		int[] connections0 = null;
		int[] connections1 = null;
		InputLayer.put(0, connections0);
		InputLayer.put(1, connections1);
		
		HashMap<Integer,int[]> Layer0 = new HashMap<Integer, int[]>();
		int[] connections2 = {0};
		int[] connections3 = {1};
		int[] connections4 = {0,1};
		Layer0.put(2, connections2);
		Layer0.put(3, connections3);
		Layer0.put(4, connections4);
		
		HashMap<Integer,int[]> OutputLayer = new HashMap<Integer, int[]>();
		int[] connections5 = {0,2,3,4};
		OutputLayer.put(5,connections5);
		
		ArrayList<HashMap<Integer,int[]>> networkStructure = new ArrayList<HashMap<Integer,int[]>>();
		networkStructure.add(InputLayer);
		networkStructure.add(Layer0);
		networkStructure.add(OutputLayer);
		
		//  ---------------   Set up the input weights ----------------------------------
		
		// setup this whole area so it's a separate class.  That way you can check his online version of 
		// the other network
		// --> so, assignment data class. and also test data class.  
		
		MLNetwork network = new MLNetwork();
		
		
	}

}
