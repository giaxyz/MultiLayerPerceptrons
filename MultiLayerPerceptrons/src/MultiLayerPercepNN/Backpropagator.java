package MultiLayerPercepNN;


public class Backpropagator {

	private MLNetwork network;
	private int networkSize;
	private boolean test1Neuron;
	private int neuronToTest;
	
	public Backpropagator(MLNetwork network) {
		
		this.network = network;
		this.networkSize = network.getNetworkBiasValues(false).size();
		this.test1Neuron = true; // set this to test only 1 layer, it will backprop from the out to the test neuron
		this.neuronToTest = 4;
		
	}

	public void backpropagate() {
		
		System.out.println("\n\n\t\t ----------------  Backpropagating ---------------  ");
		
		// Make sure that the last neuron is first set to the last actual neuron, not an input one
		int lastActualNeuron = getLastNeuronInNetwork(false);
		int lastNeuron = confirmLastNeuron(lastActualNeuron, false);
		
		// Now back propagate to the last neuron
		for(int i = (this.networkSize - 1);  i>=lastNeuron; i--){
		
			Perceptron currentNeuron = this.network.getNeurons(false).get(i);
			System.out.println("\t" + currentNeuron);
			
			double betaError = computeBetaError(currentNeuron, false);
			currentNeuron.setBetaError(betaError, false);
			System.out.println("\t---" +  " betaError set to : " + betaError + "\n");
		}
		
	}
	
	public double computeBetaError(Perceptron neuron, boolean printInfo){
		
		double tmp = 0.0;
		return tmp;
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
