package LVQ;
import java.util.Scanner;
import java.text.DecimalFormat;

public class RunLVQ {
	private static Scanner input;

	public static void main(String[] args) {
		
		double[][] weight = {
				{0.867, 3.27/4},
				{0.670, 3.39/4},
				{0.681, 3.52/4}
				};
		int[] wClass = 
				{1,2,3};
		// data init
		double[][] data = 
				{
					{0.75, 3.07/4},//1
					{0.60, 3.00/4},//1
					{0.70, 3.03/4},//1
					{0.62, 3.18/4},//1
					{0.83, 3.29/4},//1
					{0.78, 3.71/4},//1
					{0.82, 2.98/4},//1	
					{0.74, 3.79/4},//1
					{0.75, 3.47/4},//1
					{0.71, 3.58/4},//1
					{0.86, 3.56/4},//1
					{0.76, 3.70/4},//1
					{0.72, 3.29/4},//1
					{0.80, 3.40/4},//1
					{0.61, 3.84/4},//1
					{0.79, 3.60/4},//1
					{0.48, 3.57/4},//1
					{0.84, 3.43/4},//1
					{0.51, 3.85/4},//1
					{0.77, 3.37/4},//1
					{0.92, 3.67/4},//1
					{0.80, 2.88/4},//1
					{0.42, 2.82/4},//1
					{0.52, 3.80/4},//1
					{0.53, 2.90/4},//1
					{0.80, 3.45/4},//1
					{0.60, 3.60/4},//1
					{0.83, 3.38/4},//1
					{0.69, 3.01/4},//1
					{0.58, 3.72/4},//1
					{0.50, 3.33/4},//1
					{0.88, 3.39/4},//1
					{0.38, 2.93/4},//1
					{0.54, 3.55/4},//1
					{0.57, 3.79/4},//2
					{0.53, 3.78/4},//2
					{0.50, 3.08/4},//2
					{0.52, 3.60/4},//2
					{0.70, 3.00/4},//2
					{0.89, 3.26/4},//2
					{0.72, 3.34/4},//2
					{0.84, 3.50/4},//2
					{0.53, 3.58/4},//2
					{0.70, 3.75/4},//2
					{0.58, 3.56/4},//2
					{0.72, 3.14/4},//2
					{0.71, 3.56/4},//2
					{0.57, 3.56/4},//2
					{0.70, 3.55/4},//2
					{0.78, 3.15/4},//2
					{0.84, 3.10/4},//2
					{0.89, 3.37/4},//2
					{0.61, 3.64/4},//2
					{0.54, 3.10/4},//2
					{0.78, 3.70/4},//2
					{0.80, 3.40/4},//2
					{0.58, 3.20/4},//2
					{0.53, 3.26/4},//2
					{0.52, 3.21/4},//2
					{0.45, 3.10/4},//2
					{0.71, 3.08/4},//2
					{0.87, 3.38/4},//2
					{0.76, 3.57/4},//2
					{0.81, 3.57/4},//3
					{0.87, 3.55/4},//3					
					{0.85, 3.70/4},//3
					{0.50, 3.51/4},//3				
					{0.50, 3.75/4},//3					
					{0.51, 3.33/4},//3
					{0.42, 3.79/4},//3					
					{0.74, 3.21/4},//3
					{0.81, 3.50/4},//3					
					{0.80, 3.29/4},//3					
				};
		
		int[] target = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
				2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3
		};
		System.out.println(target.length);
		LVQ coba = new LVQ(weight, wClass, target, data, 1000, 0.0001, 0.05);
		System.out.println("Before Learning");
		coba.printWeight();
		coba.printData();
		System.out.println("\nLearning on process........");
		coba.learn();
		System.out.println("\nAfter Learning");
		coba.printWeight();

		// test data inputan	
		//////////////////////////////////
		input = new Scanner(System.in);
		System.out.println("=========");
		double[] par = new double[4];
		for (int i=0; i < 2; i++) {
			System.out.println("masukkan parameter ke " + (i+1));
	        par[i] = input.nextDouble();	        	        
		}
			
		System.out.println("\n\nPercobaan data: {"+ par[0] +" , "+ par[1]+"}");
		double[] dataTest = {(par[0]/100),(par[1]/4)};
		coba.test(dataTest);
	}
}

class LVQ{
	// Properties needed
	double weight[][];
	double alpa;
	double input[][];
	int[] target;
	int[] wClass;
	int epoch;
	double eps; //error minimum yang diharapkan
	
	public LVQ (double[][] WEIGHT, int[] WCLASS, int[] TARGET, double[][] INPUT, int EPOCH, double EPS, double ALPA) {
		this.weight = WEIGHT;
		this.alpa = ALPA;
		this.input = INPUT;
		this.target = TARGET;
		this.wClass = WCLASS;
		this.epoch = EPOCH;
		this.eps = EPS;
	}
		
	//Train data
	void learn() {
		double[] detCheck = new double[weight.length];
		for (int i = 0; i < weight.length; i++) {
			detCheck[i] = 0;		
		}
		
		// do learning
		double errorCount=0;
		double Count = 0;
		for (int e = 0; e < this.epoch; e++) {
			System.out.println("Epoch "+(e+1)+":");
			for (int i = 0; i < this.input.length; i++) {
				// get sum of the dataVector[i] for all data
				Count++;
				for (int j = 0; j < this.input[i].length; j++) {
					for (int n = 0; n < weight.length; n++) {
						detCheck[n] = detCheck[n] + Math.pow(input[i][j]-weight[n][j],2);
					}
				}

				// do sqrt for all sum to get real result of
				// lenght/determinant checker
				for (int n = 0; n < detCheck.length; n++) {
					detCheck[n] = Math.pow(detCheck[n],0.5);
				}

				// looking for minimum length of 
				// length/determinant checker to dataVector[i]
				double min = Double.MAX_VALUE;
				int detIndex = 0;
				for (int n = 0; n < detCheck.length; n++) {
					if (min > detCheck[n]) {
						min = detCheck[n];
						detIndex = n;
					}
				}
				System.out.println("Target data "+ (i+1) +": "+target[i]);
				System.out.println("Data "+ (i+1) +" Nearest to " + (detIndex+1));
				
				if (target[i] == (detIndex+1)) {
				// Update Weight based on minimum length to the data
				for (int j = 0; j < this.weight[detIndex].length; j++) {
					weight[detIndex][j] = weight[detIndex][j] + alpa*(input[i][j]-weight[detIndex][j]);
					}
				} 	
				
				else { 
					errorCount++;
					for (int j = 0; j < this.weight[detIndex].length; j++) {
						weight[detIndex][j] = weight[detIndex][j] - alpa*(input[i][j]-weight[detIndex][j]);					
				}
			}
			}
			alpa = eps*alpa;
			System.out.println();
			
		}	
		double acc = errorCount/Count;
			System.out.println("Total data iterasi: " + Count);
			System.out.println("Total data benar: " + errorCount);
			System.out.println("Total data error: " + (Count - errorCount));
			DecimalFormat df = new DecimalFormat("#.###");
			System.out.println("Akurasi: " + df.format(acc*100) + "%");
	
	}//learn
	
	/* Test for data test*/
	void test(double[] dataTest){
		// length/determinant checker init 
		double[] detCheck = new double[weight.length];
		for (int i = 0; i < weight.length; i++) {
			detCheck[i] = 0;
		}

		// get sum of the dataVector of dataTest
		for (int j = 0; j < dataTest.length; j++) {
			for (int n = 0; n < weight.length; n++) {
				detCheck[n] = detCheck[n] + Math.pow(dataTest[j]-weight[n][j],2);
			}
		}
		// do sqrt for all detCheck
		for (int n = 0; n < detCheck.length; n++) {
			detCheck[n] = Math.pow(detCheck[n],0.5);
		}

		double min = Double.MAX_VALUE;
		int detIndex = 0;
		for (int n = 0; n < detCheck.length; n++) {
			if (min > detCheck[n]) {
				min = detCheck[n];
				detIndex = n;
			}
		}
		
		// print length result
		for (int i = 0; i < detCheck.length; i++) {
			System.out.print(detCheck[i]+"\t");
		}
		System.out.println();
		System.out.print("Data is in class: ");
		if(detIndex+1==1) {
		System.out.println("SNMPTN");}
		else if(detIndex+1==2) {
			System.out.println("SBMPTN");}
		else if(detIndex+1==3) {
			System.out.println("PENMABA");}	
	}//Test()

	void printWeight(){
		System.out.println("==============");
		System.out.println("Weight :");
		for (int i = 0; i < this.weight.length; i++) {
			for (int j = 0; j < this.weight[i].length; j++) {
				System.out.print(this.weight[i][j]+"\t");
			}
		}System.out.println();
	}

	void printData(){
		System.out.println("==============");
		System.out.println("Data sample :");
		for (int i = 0; i < this.input.length; i++) {
			for (int j = 0; j < this.input[i].length; j++) {
				System.out.print(this.input[i][j]+"\t");
			}
			System.out.println();
		}
	}//printData()
}//LVQ()





