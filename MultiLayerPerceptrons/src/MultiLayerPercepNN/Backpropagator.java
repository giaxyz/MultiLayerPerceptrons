package MultiLayerPercepNN;

import java.util.ArrayList;


public class Backpropagator {

	private MLNetwork network;
	private int networkSize;
	private boolean test1Neuron;
	private int neuronToTest;
	private int currentExample;
	private boolean printBackPropInfo;
	
	public Backpropagator(MLNetwork network) {
		
		this.network = network;
		this.networkSize = network.getNetworkBiasValues(false).size();
		this.test1Neuron = true; // set this to test only 1 layer, it will backprop from the out to the test neuron
		this.neuronToTest = 4;
		this.currentExample = 0;
		this.printBackPropInfo = true;
		
	}

	public void backpropagate(int exampleNumber) {
		
		System.out.println("\n\n\t\t ----------------  Backpropagating ---------------  ");
		
		// Make sure that the last neuron is first set to the last actual neuron, not an input one
		int lastActualNeuron = getLastNeuronInNetwork(false);
		int lastNeuron = confirmLastNeuron(lastActualNeuron, false);
		
		//Set the current Example
		setCurrentExample(exampleNumber);
		
		// Now back propagate to the last neuron
		for(int i = (this.networkSize - 1);  i>=lastNeuron; i--){
		
			Perceptron currentNeuron = this.network.getNeurons(false).get(i);
			
			if(printBackPropInfo){
				System.out.println("-- Neuron " + currentNeuron + "-- : ");
			}
			
			
			double betaError = computeBetaError(currentNeuron, false);
			currentNeuron.setBetaError(betaError, false);
			
			if(printBackPropInfo){
				System.out.println("\t---" +  " betaError set to : " + betaError + "\n");
			}
			
			
		}
		
	}
	
	public double computeBetaError(Perceptron neuron, boolean printInfo){
		
		
		double betaError = Double.NEGATIVE_INFINITY;
		double derivative = Double.NEGATIVE_INFINITY;
		double out = neuron.getOutput(printBackPropInfo);
		int currentNeuronID = neuron.getNeuronID(false);
		
		if(neuron.getIsOutput()){
		
			ArrayList<Double> outputsY = network.getOutputData(false);
			double expectedY = outputsY.get(currentExample);
			
			derivative = (expectedY - out);
			
			if(printBackPropInfo){
				
				System.out.println("\t" + neuron + " is an output neuron");
				//System.out.println("\tExpected Y : " + expectedY);
				System.out.println("\tDerivative : " + derivative);
			}
			
		}else if(neuron.getIsInput()){
			
			System.out.println(neuron + " is input. Skipping Beta Error");
			
			
		}else{
			
			System.out.println(neuron + " is normal");
		}
		
		
		
		betaError = MlUtils.formatDouble((out * ( 1 - out ) * derivative));
		
		if(printInfo){
			System.out.println("\t" + neuron + " beta error set to : " + betaError);
		}
		
		return betaError;
		
	}
	
	public int getCurrentExample(boolean printInfo){
		
		if(printInfo){
			System.out.println("Current Example Number is : " + this.currentExample);
		}
		return this.currentExample;
		
	}
	
	private void setCurrentExample(int exampleNumber){
		this.currentExample = exampleNumber;
	}

	private int getLastNeuronInNetwork(boolean printInfo){
		
		int lastActualNeuron = 0;
		
		for(int i = 0; i<this.networkSize; i++){
			
			Perceptron currentNeuron = this.network.getNeurons(false).get(i);
			boolean isInput = currentNeuron.getIsInput();
			if(printInfo){
				System.out.println(currentNeuron + " isInput : " + isInput);
			}
			if(!isInput){
				lastActualNeuron = i;
				break;
			}
		}
		
		if(printInfo){
			
			System.out.println("Last actual neuron : " + lastActualNeuron);
		}
		
		return lastActualNeuron;
	}

	private int confirmLastNeuron(int lastActualNeuron, boolean printInfo){
		
	int lastNeuron = lastActualNeuron;
		
		if((test1Neuron) && (this.neuronToTest >= lastNeuron)){
			lastNeuron = this.neuronToTest;
		}
		
		if(printInfo){
			System.out.println("Last neuron set to : " + lastNeuron);
		}
		
		
		return lastNeuron;
	}
}
