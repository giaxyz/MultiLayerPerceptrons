package MultiLayerPercepNN;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MlUtils {
	
	public static void printDoubleArray(double[] doubleArray){
		
		System.out.print("[");
		if(doubleArray != null){
			for(int i = 0; i< doubleArray.length; i++){
				System.out.print(" " + doubleArray[i] + " ");
			}
			
		}else{
			System.out.print("null");
		}
		
		System.out.print("]");	
	}
	
	public static void printIntArray(int[] intArray){
		
		System.out.print("[");
		if(intArray != null){
			for(int i = 0; i< intArray.length; i++){
				System.out.print(" " + intArray[i] + " ");
			}
			
		}else{
			System.out.print("null");
		}
		
		System.out.print("]");	
	}

	public static void printHashMapIntIntArray(HashMap<Integer, int[]> map) {
		
		Iterator<Map.Entry<Integer, int[]>> iterator = map.entrySet().iterator() ;
        while(iterator.hasNext()){
        	
            Map.Entry<Integer, int[]> mapEnt = iterator.next();
            
            
            int[] value = mapEnt.getValue();
            int key = mapEnt.getKey();
            
            System.out.print("Neuron : " + key + ": [");
            
            if(value != null){
            	for(int i = 0; i<value.length; i++){
            		System.out.print( " " + value[i] + ",");
            		
            	}
            }else{
            	System.out.print("null");
            }
            
            System.out.print("]\n");
        }
       
	}

	public static void printHashMapIntDoubleArray(HashMap<Integer, double[]> map) {
		
		Iterator<Map.Entry<Integer, double[]>> iterator = map.entrySet().iterator() ;
        while(iterator.hasNext()){
        	
            Map.Entry<Integer, double[]> mapEnt = iterator.next();
            
            
            double[] value = mapEnt.getValue();
            int key = mapEnt.getKey();
            
            System.out.print("Neuron : " + key + ": [");
            
            if(value != null){
            	for(int i = 0; i<value.length; i++){
            		System.out.print( " " + value[i] + ",");
            		
            	}
            }else{
            	System.out.print("null");
            }
            
            System.out.print("]\n");
        }
       
	}
	
	public static void printHashMapIntDouble(HashMap<Integer, Double> map) {
		
	
	
        Iterator<Map.Entry<Integer, Double>> iterator = map.entrySet().iterator() ;
        while(iterator.hasNext()){
            Map.Entry<Integer, Double> mapEnt = iterator.next();
            System.out.println(mapEnt.getKey() +" : "+ mapEnt.getValue());
        }
       
	}

	public static ArrayList<Double> duplicateArrayList(ArrayList<Double> originalList){
		
		ArrayList<Double> newList = new ArrayList<Double>();
		for(int i = 0; i< originalList.size(); i++){
			newList.add(originalList.get(i));
		}
		
		return newList;
		
	}
}
