package MultiLayerPercepNN;

public class MLNetwork {

	public MLNetwork(){
		
		boolean printInfo = true;
		boolean isOutput = false;
		boolean isInput = false;
		double biasValue = 1.0;
		double biasWeight = 0.0;
		int neuronID =  0;
		int layerID = 1; 
		Perceptron perceptron = createPerceptron(biasWeight, biasValue, neuronID, layerID, isOutput, isInput, printInfo);
		perceptron.getBetaError(true);
		perceptron.getInputIDs(true);
	}
	
	
	// Create A Perceptron
	public Perceptron createPerceptron(double biasWeight, double biasValue, int neuronID, int layerID, boolean isOutput, boolean isInput, boolean printInfo) {
		
		
		Perceptron perceptron = new Perceptron(biasWeight, biasValue, neuronID, layerID, isOutput, isInput);
		if(printInfo){
			
			System.out.println("Perceptron Details : " + perceptron);
		}
		return perceptron;
		
	}
}
