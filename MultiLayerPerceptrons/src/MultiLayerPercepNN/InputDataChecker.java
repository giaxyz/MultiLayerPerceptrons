package MultiLayerPercepNN;

import java.util.ArrayList;

public class InputDataChecker {

	ArrayList<ArrayList<Double>> data;

	
	public InputDataChecker(ArrayList<ArrayList<Double>> data){
		
		this.data = data;
	}

	public boolean setData(ArrayList<ArrayList<Double>> data){
		
		this.data = data;
		return true;
	}
	
	public ArrayList<ArrayList<Double>> getData(boolean printInfo){
		
		if(printInfo){
			System.out.println("Input Data is : " + this.data);
		}
		return this.data;
	}

	public void checkData() throws Exception {
		
		int firstDataSize = this.data.get(0).size();
		
		for(int i = 0; i<this.data.size(); i++){
			
			int dataSize = this.data.get(i).size();
			
			if(dataSize != firstDataSize){
				throw new Exception("Error : Input Data array sizes mismatch");
			}
			
		}
		
	}
}
