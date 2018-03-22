import java.util.*;
import java.io.File;

class ANN
{
    boolean verbose = true;

    int i = 64; // number of input neurons
    int j = 5; // number of hidden neurons
    int k = 10; // number of output neurons
    double[][] weights1 = new double[j][i];
    double[][] weights2 = new double[k][j];

    ArrayList<Double> delta_j;
    ArrayList<Double> delta_k;

    ArrayList<Double> list_zi;
    ArrayList<Double> list_aj;
    ArrayList<Double> list_zj;
    ArrayList<Double> list_ak;
    ArrayList<Double> list_yk;

    double[][] derivatives_E_Wji = new double[j][i];
    double[][] derivatives_E_Wkj = new double[k][j];

    ANN()
    {
        // Initialize weight matrices based on Gaussian distribution
        Random r = new Random();
        for (int j = 0; j < this.j; j++)
        {
            for (int i = 0; i < this.i; i++)
            {
                weights1[j][i] = r.nextGaussian();
            }
        }
        for (int k = 0; k < this.k; k++)
        {
            for (int j = 0; j < this.j; j++)
            {
                weights2[k][j] = r.nextGaussian();
            }
        }
    }

    void train(ArrayList<Example> train_x, int max_iterations, ArrayList<Example> validation_x)
    {
        int iterations = 0;
        double error = 0;
        double old_validation_error = Double.POSITIVE_INFINITY;
        while(iterations++ < max_iterations)
        {
            error = 0;
            // Here we're doing sample by sample gradient descent (not mini-batch)
            for (Example xi: train_x)
            {
                ArrayList<Double> y = this.feedForward(xi.attributes); // y.size() = k

                double error_n = 0;
                for (int k = 0; k < this.k; k++)
                {
                    error_n += (0.5)*Math.pow(y.get(k) - xi.target.get(k) ,2);
                }
                error_n /= this.k;
                error += error_n;

                delta_k = this.evaluateOutputDeltas(y, xi.target);

                delta_j = this.backprop(this.list_aj, this.weights2, delta_k);

                this.evaluateDerivatives(); // 2 matrices of derivatives

                this.gradientDescent(weights1, weights2, derivatives_E_Wji, derivatives_E_Wkj);
            }
            error /= train_x.size();

            double validation_error = getValidationError(validation_x);
            System.out.println("[" + iterations + "] Train error: " + error);
            System.out.println("[" + iterations + "] Validation error: " + validation_error);
            System.out.println();
            if (validation_error > old_validation_error)
                break;
            old_validation_error = validation_error;
        }
        System.out.println("--------------");
        System.out.println("Train error: " + error);
    }

    // Perform gradient descent on weight matrix w by iterating over samples in x with learning rate eta
    // Iterate until error < epsilon or number of iterations > max_iterations
    void gradientDescent(double[][] weights1, double[][] weights2, double[][] D_Wji, double[][] D_Wkj)
    {
        // Hyper-parameters
        double eta = 0.01;

        for (int k = 0; k < this.k; k++)
        for (int j = 0; j < this.j; j++)
        weights2[k][j] -= eta * D_Wkj[k][j];

        for (int j = 0; j < this.j; j++)
        for (int i = 0; i < this.i; i++)
        weights1[j][i] -= eta * D_Wji[j][i];

    }

    void evaluateDerivatives()
    {
        for (int j = 0; j < this.j; j++)
        {
            for (int i = 0; i < this.i; i++)
            {
                derivatives_E_Wji[j][i] = delta_j.get(j) * list_zi.get(i);
            }
        }

        for (int k = 0; k < this.k; k++)
        {
            for (int j = 0; j < this.j; j++)
            {
                derivatives_E_Wkj[k][j] = delta_k.get(k) * list_zj.get(j);
            }
        }
    }

    ArrayList<Double> backprop(ArrayList<Double>list_aj, double[][] weights_kj, ArrayList<Double> delta_k)
    {
        ArrayList<Double> delta_j = new ArrayList<Double>();

        // Implementation here
        for (int j = 0; j < this.j; j++)
        {
            double sum = 0;
            for (int k = 0; k < this.k; k++)
            {
                sum += weights_kj[k][j] * delta_k.get(k);
            }
            delta_j.add(sum * delta_sigmoid(list_aj.get(j)));
        }

        return delta_j;
    }

    ArrayList<Double> evaluateOutputDeltas(ArrayList<Double> y, ArrayList<Integer> target)
    {
        ArrayList<Double> delta_k = new ArrayList<Double>();
        for (Double yk: y)
        for (int k = 0; k < y.size(); k++)
        {
            delta_k.add((y.get(k) - target.get(k)) * delta_sigmoid(list_ak.get(k)));
        }
        return delta_k;
    }

    // Return list of output values
    ArrayList<Double> feedForward(ArrayList<Double> x)
    {
        list_zi = new ArrayList<Double>();
        for(int i = 0; i < this.i; i++)
        {
            list_zi.add(x.get(i));
        }

        list_aj = new ArrayList<Double>();
        list_zj = new ArrayList<Double>();
        for (int j = 0; j < this.j; j++)
        {
            double aj = 0;
            for (int i = 0; i < this.i; i++)
            {
                aj += weights1[j][i] * list_zi.get(i);
            }
            list_aj.add(aj);
            list_zj.add(sigmoid(aj));
        }

        list_ak = new ArrayList<Double>();
        list_yk = new ArrayList<Double>();
        for (int k = 0; k < this.k; k++)
        {
            double ak = 0;
            for (int j = 0; j < this.j; j++)
            {
                ak += weights2[k][j] * list_zj.get(j);
            }
            list_ak.add(ak);
            list_yk.add(sigmoid(ak));
        }

        return list_yk;
    }

    double getValidationError(ArrayList<Example> validation_x)
    {
        double validation_error = 0;
        for (Example xi: validation_x)
        {
            ArrayList<Double> y = this.feedForward(xi.attributes);
            double En = 0;
            for (int k = 0; k < this.k; k++)
            {
                En += (0.5)*Math.pow(y.get(k) - xi.target.get(k) ,2);
            }
            En /= this.k;
            validation_error += En;
        }
        validation_error /= validation_x.size();
        return validation_error;
    }

    void evaluateModel(ArrayList<Example> test_x)
    {
        Utility uObj = new Utility(); // Utility object

        int FP = 0, FN = 0, TP = 0, TN = 0; // Initialize variables
        double test_error = 0;

        for (Example xi: test_x)
        {
            ArrayList<Double> y = this.feedForward(xi.attributes);
            ArrayList<Integer> prediction = new ArrayList<Integer>();
            double En = 0;
            for (int k = 0; k < this.k; k++)
            {
                En += (0.5)*Math.pow(y.get(k) - xi.target.get(k) ,2);
                int p = (int) Math.round(y.get(k));
                prediction.add(p);
            }
            En /= this.k;
            test_error += En;

            // TODO: Makeshift evaluation
            // if (prediction.get(0) == 0 && xi.target.get(0) == 0)
            //     TP++;
            // if (prediction.get(0) == 1 && xi.target.get(0) == 1)
            //     TN++;
            // if (prediction.get(0) == 0 && xi.target.get(0) == 1)
            //     FN++;
            // if (prediction.get(0) == 1 && xi.target.get(0) == 0)
            //     FP++;

        }
        test_error /= test_x.size();
        System.out.println("Test error: " + test_error);

        // /* Print confusion matrix */
        // uObj.computeConfusionMatrix(TP, FP, TN, FN);
        //
        // /* Compute precision */
        // uObj.computePrecision(TP, FP, TN, FN);
        //
        // /* Compute recall */
        // uObj.computeRecall(TP, FP, TN, FN);
    }

    static double delta_sigmoid(double x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    static double sigmoid(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }

    public static void main(String[] args)
    {
        Utility uObj = new Utility(); // Utility object
        ANN ann = new ANN();

        String train_data = "";
        String test_data = "";
        String validation_data = "";

        try {
            train_data = args[0];
            test_data = args[1];
            validation_data = args[2];

            String options;
            try {
                options = args[3];
                if (options.equals("-v") || options.equals("--verbose"))
                    ann.verbose = true;
                if (options.equals("-h") || options.equals("--help"))
                {
                    // showHelp(); TODO: Add help
                    System.exit(0);
                }
            }catch(Exception e){}
        } catch(Exception e) {
            // showHelp(); TODO: Add help
            System.exit(0);
        }

        ArrayList<Example> x = new ArrayList<Example>(); // x matrix. Each data point is of type Example
        ArrayList<Example> test_x = new ArrayList<Example>(); // Test data points
        ArrayList<Example> validation_x = new ArrayList<Example>();


        try {
            x = uObj.readInputCSV(train_data);
            test_x = uObj.readInputCSV(test_data);
            validation_x = uObj.readInputCSV(validation_data);
        }
        catch(Exception e) {
            System.out.println("File not found!");
            System.exit(0);
        }

        ann.train(x, 10000, validation_x); // training data: x, max_iterations: 100
        ann.evaluateModel(test_x);

    }
}
