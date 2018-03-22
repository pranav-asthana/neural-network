import java.util.*;
import java.io.File;

class Utility
{
	/** Parses a CSV file and stores info in transactions
	*/
	static ArrayList<Example> readInputCSV(String fileName) throws Exception
	{
		File file = new File(fileName);
		Scanner sc = new Scanner(file);
		ArrayList<Example> data  = new ArrayList<Example>();

		/* Converting each line into a transaction */
		while(sc.hasNextLine()){
			String line = sc.nextLine();
			String[] values = line.split(",");
			ArrayList<Double> a = new ArrayList<Double>();
			for(int i = 0; i < 64; i++)
				a.add(Double.parseDouble(values[i]));
			int t = Integer.parseInt(values[64]);

			Example temp = new Example(a, t);

			data.add(temp);
		}
		return data;
	}

	/* Print the confusion matrix */
	static void computeConfusionMatrix(int TP, int FP, int TN, int FN)
	{
		// P (positive) -> class 0
		// N (negative) -> class 1
		System.out.println("\n\n\t\tActual Class\tTotal");
		System.out.println("Predicted Class\tC0\tC1");
		System.out.println("C0\t\t"+TN+"\t"+FP+"\t"+(TN+FP));
		System.out.println("C1\t\t"+FN+"\t"+TP+"\t"+(FN+TP));
        System.out.println("Total\t\t"+(TN+FN)+"\t"+(FP+TP)+"\n");


	}

	/* Compute precision */
	static void computePrecision(int TP,int FP, int TN, int FN)
	{
		float x = (float)100*TP/(TP+FP);
		float y = (float)100*TN/(TN+FN);

		System.out.println("Precision for class 0: " + x);
		System.out.println("Precision for class 1: " + y + "\n");
	}

	/* Compute recall */
	static void computeRecall(int TP, int FP, int TN, int FN)
	{
		float x = (float)100*TP/(TP+FN);
		float y = (float)100*TN/(TN+FP);

		System.out.println("Recall for class 0: " + x);
		System.out.println("Recall for class 1: " + y + "\n");
	}
}
