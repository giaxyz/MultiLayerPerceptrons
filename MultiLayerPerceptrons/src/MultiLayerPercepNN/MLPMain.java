package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.Arrays;

public class MLPMain {

	public static void main(String[] args) throws Exception {
		
		// ------------ Set up to Read all Data -------------------------------
		
		ArrayList<ArrayList<Double>> data = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> inputsX1 =  new ArrayList<Double>(Arrays.asList(1.0,1.0));
		ArrayList<Double> inputsX2 =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		ArrayList<Double> outputs =  new ArrayList<Double>(Arrays.asList(0.0,1.0));
		data.add(inputsX1);
		data.add(inputsX2);
		data.add(outputs);
		InputDataChecker inputData = new InputDataChecker(data);
		inputData.checkData();
		inputData.getData(true);
		
		// think about setting up the data structure
		MLNetwork network = new MLNetwork();
		
		
	}

}
