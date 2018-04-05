import java.util.*;
import java.io.File;

class ANN
{
    boolean verbose = false;

    int i = 64; // number of input neurons
    int j; // number of hidden neurons
    int k = 10; // number of output neurons
    double[][] weights1 = new double[j][i+1];
    double[][] weights2 = new double[k][j+1];

    // Momentum matrices for the 2 layers
    double[][] M1 = new double[j][i+1];
    double[][] M2 = new double[k][j+1];

    // RMS matrices for the 2 layers
    double[][] R1 = new double[j][i+1];
    double[][] R2 = new double[k][j+1];

    // Used for backpropagation
    ArrayList<Double> delta_j;
    ArrayList<Double> delta_k;

    // Lists to store intermediate outputs
    ArrayList<Double> list_zi;
    ArrayList<Double> list_aj;
    ArrayList<Double> list_zj;
    ArrayList<Double> list_ak;
    ArrayList<Double> list_yk;

    // Derivatives used for backpropagation
    double[][] derivatives_E_Wji = new double[j][i+1];
    double[][] derivatives_E_Wkj = new double[k][j+1];

    // Error fucntion used
    boolean crossEntropy = true; // false -> Sum of Squared Errors

    ANN(int hidden, boolean verbose) // hidden = number of hidden neurons
    {
        j = hidden;
        this.verbose = verbose;

        weights1 = new double[j][i+1];
        weights2 = new double[k][j+1];

        M1 = new double[j][i+1];
        M2 = new double[k][j+1];

        R1 = new double[j][i+1];
        R2 = new double[k][j+1];

        derivatives_E_Wji = new double[j][i+1];
        derivatives_E_Wkj = new double[k][j+1];

        // Initialize weight matrices based on Gaussian distribution with mean 0, variance 1
        // Initialize momentum and rms matrices to 0
        Random r = new Random();
        for (int j = 0; j < this.j; j++)
        {
            for (int i = 0; i <= this.i; i++)
            {
                weights1[j][i] = r.nextGaussian();
                M1[j][i] = 0;
                R1[j][i] = 0;
            }
        }
        for (int k = 0; k < this.k; k++)
        {
            for (int j = 0; j <= this.j; j++)
            {
                weights2[k][j] = r.nextGaussian();
                M2[k][j] = 0;
                R2[k][j] = 0;
            }
        }
    }

    int train(ArrayList<Example> train_x, int batch_size, int max_iterations, ArrayList<Example> validation_x)
    {
        int iterations = 0;
        double error = 0;
        double old_validation_error = Double.POSITIVE_INFINITY;
        int spike = 0;
        while(iterations++ < max_iterations)
        {
            error = 0;
            ArrayList<ArrayList<Example>> batches = generateBatches(train_x, batch_size); // Get batches
            for (ArrayList<Example> batch: batches) // For each batch
            {
                Matrix sum_derivatives_E_Wji = new Matrix(new double[j][i+1]);
                Matrix sum_derivatives_E_Wkj = new Matrix(new double[k][j+1]);
                for (Example xi: batch) // For each sample
                {
                    // Get the outputs with the current weights
                    ArrayList<Double> y = this.feedForward(xi.attributes); // y.size() = k

                    // Calculate error for the sample
                    double error_n = 0;
                    for (int k = 0; k < this.k; k++)
                    {
                        if (crossEntropy)
                            error_n += -(xi.target.get(k) * Math.log(y.get(k)));
                        else
                            error_n += (0.5)*Math.pow(y.get(k) - xi.target.get(k) ,2);
                    }
                    error_n /= this.k; // Normalize based on number of output classes
                    error += error_n; // Add to total error

                    delta_k = this.evaluateOutputDeltas(y, xi.target); // Calculate output delta values


                    delta_j = this.backprop(this.list_aj, this.weights2, delta_k); // Calculate hidden delta values using backpropagation
                    this.evaluateDerivatives(); // 2 matrices of derivatives
                    // Sum derivatives for batch processing
                    sum_derivatives_E_Wji.add(new Matrix(derivatives_E_Wji));
                    sum_derivatives_E_Wkj.add(new Matrix(derivatives_E_Wkj));
                }
                // Perform gradient descent with the batch error derivatives
                this.gradientDescent(weights1, weights2, M1, M2, sum_derivatives_E_Wji.matrix, sum_derivatives_E_Wkj.matrix, iterations);
            }
            error /= train_x.size(); // Normalize total error based on number of samples

            double validation_error = getValidationError(validation_x);
            if (verbose)
            {
                System.out.printf("[" + iterations + "] Train error: %.10f", error);
                System.out.printf(" Validation error: %.10f\n", validation_error);
            }
            if ((validation_error - old_validation_error) > 0.01*old_validation_error)
                spike++;
            if (spike == 5) // Stop training when validation_error increases for the 5th time
                break;
            old_validation_error = validation_error;
        }
        if (verbose)
        {
            System.out.println("--------------");
            System.out.println("[" + iterations + "] Train error: " + error);
        }
        return iterations;
    }

    ArrayList<ArrayList<Example>> generateBatches(ArrayList<Example> train_x, int batch_size)
    {
        ArrayList<ArrayList<Example>> batches = new ArrayList<ArrayList<Example>>();
        Collections.shuffle(train_x); // Randomly pick `batch_size` number of samples
        for (int i = 0; i < train_x.size(); i += batch_size)
        {
            if (i+batch_size < train_x.size()) // Last batch may contain less samples
                batches.add(new ArrayList(train_x.subList(i, i+batch_size)));
            else
                batches.add(new ArrayList(train_x.subList(i, train_x.size())));
        }
        return batches;
    }

    // Perform gradient descent on weight matrix w by iterating over samples in x with learning rate eta
    // The adam optimizer has been implemented
    void gradientDescent(double[][] weights1, double[][] weights2, double[][] M1, double[][] M2, double[][] D_Wji, double[][] D_Wkj, int iteration)
    {
        // Hyper-parameters
        double eta = 0.01;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 10e-8;

        for (int k = 0; k < this.k; k++)
        {
            for (int j = 0; j <= this.j; j++)
            {
                M2[k][j] = beta1*M2[k][j] + (1-beta1)*D_Wkj[k][j];
                R2[k][j] = beta2*R2[k][j] + (1-beta2)*Math.pow(D_Wkj[k][j], 2);
                double M2_corrected = M2[k][j] / (1 - Math.pow(beta1, iteration));
                double R2_corrected = R2[k][j] / (1 - Math.pow(beta2, iteration));
                weights2[k][j] -= eta * (M2_corrected / (Math.sqrt(R2_corrected) + epsilon));

                // weights2[k][j] -= eta * D_Wkj[k][j]; // Simple gradient descent
            }
        }

        for (int j = 0; j < this.j; j++)
        {
            for (int i = 0; i <= this.i; i++)
            {
                M1[j][i] = beta1*M1[j][i] + (1-beta1)*D_Wji[j][i];
                R1[j][i] = beta2*R1[j][i] + (1-beta2)*Math.pow(D_Wji[j][i], 2);
                double M1_corrected = M1[j][i] / (1 - Math.pow(beta1, iteration));
                double R1_corrected = R1[j][i] / (1 - Math.pow(beta2, iteration));
                weights1[j][i] -= eta * (M1_corrected / (Math.sqrt(R1_corrected) + epsilon));

                // weights1[j][i] -= eta * D_Wji[j][i]; // Simple gradient descent
            }
        }

    }

    void evaluateDerivatives()
    {
        for (int j = 0; j < this.j; j++)
        {
            for (int i = 0; i <= this.i; i++)
            {
                derivatives_E_Wji[j][i] = delta_j.get(j) * list_zi.get(i);
            }
        }

        for (int k = 0; k < this.k; k++)
        {
            for (int j = 0; j <= this.j; j++)
            {
                derivatives_E_Wkj[k][j] = delta_k.get(k) * list_zj.get(j);
            }
        }
    }

    ArrayList<Double> backprop(ArrayList<Double>list_aj, double[][] weights_kj, ArrayList<Double> delta_k)
    {
        ArrayList<Double> delta_j = new ArrayList<Double>();

        for (int j = 0; j <= this.j; j++)
        {
            double sum = 0;
            for (int k = 0; k < this.k; k++)
            {
                sum += weights_kj[k][j] * delta_k.get(k);
            }
            if (j == this.j)
                delta_j.add(0.0d); // For bias term
            else
                delta_j.add(sum * delta_sigmoid(list_aj.get(j)));
        }

        return delta_j;
    }

    ArrayList<Double> evaluateOutputDeltas(ArrayList<Double> y, ArrayList<Integer> target)
    {
        ArrayList<Double> delta_k = new ArrayList<Double>();
        for (int k = 0; k < y.size(); k++)
        {
            if (crossEntropy)
                delta_k.add((y.get(k) - target.get(k)));
            else
                delta_k.add((y.get(k) - target.get(k)) * delta_sigmoid(list_ak.get(k)));
        }
        return delta_k;
    }

    // Return list of output values
    ArrayList<Double> feedForward(ArrayList<Double> x)
    {
        list_zi = new ArrayList<Double>();
        list_zi.add(1.0d); // x(0) = 1, for bias. b = Wj0
        for(int i = 0; i < this.i; i++)
        {
            list_zi.add(x.get(i));
        }

        list_aj = new ArrayList<Double>();
        list_zj = new ArrayList<Double>();
        list_zj.add(1.0d); // for bias to next layer; Wk0
        for (int j = 0; j < this.j; j++)
        {
            double aj = 0;
            for (int i = 0; i <= this.i; i++)
            {
                aj += weights1[j][i] * list_zi.get(i);
            }
            list_aj.add(aj);
            list_zj.add(sigmoid(aj));
        }

        list_ak = new ArrayList<Double>();
        list_yk = new ArrayList<Double>();
        double softmax_denominator = 0;
        double C = 100; // To stabilize softmax
        for (int k = 0; k < this.k; k++)
        {
            double ak = 0;
            for (int j = 0; j <= this.j; j++)
            {
                ak += weights2[k][j] * list_zj.get(j);
            }
            list_ak.add(ak);
            if (crossEntropy)
            {
                list_yk.add(Math.exp(ak + Math.log(C)));
                softmax_denominator += Math.exp(ak + Math.log(C));
            }
            else
                list_yk.add(sigmoid(ak));
        }
        if (crossEntropy)
        {
            for (int index = 0; index < list_yk.size(); index++)
            {
                list_yk.set(index, list_yk.get(index)/softmax_denominator);
            }
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
                if (crossEntropy)
                    En += -(xi.target.get(k) * Math.log(y.get(k)));
                else
                    En += (0.5)*Math.pow(y.get(k) - xi.target.get(k) ,2);
            }
            En /= this.k;
            validation_error += En;
        }
        validation_error /= validation_x.size();
        return validation_error;
    }

    double evaluateModel(ArrayList<Example> test_x)
    {
        Utility uObj = new Utility(); // Utility object

        int correct = 0, incorrect = 0;
        double test_error = 0;

        for (Example xi: test_x)
        {
            ArrayList<Double> y = this.feedForward(xi.attributes);
            double En = 0;
            int p = -1;
            double max_prob = 0;
            for (int k = 0; k < this.k; k++)
            {
                if (crossEntropy)
                    En += -(xi.target.get(k) * Math.log(y.get(k)));
                else
                    En += (0.5)*Math.pow(y.get(k) - xi.target.get(k) ,2);
                if (y.get(k) > max_prob)
                {
                    p = k;
                    max_prob = y.get(k);
                }
            }
            En /= this.k;
            test_error += En;

            if (p == xi.number)
                correct++;
            else
                incorrect++;
        }
        test_error /= test_x.size();
        double accuracy = (float)correct*100 / test_x.size();

        if (verbose)
        {
            System.out.println("Test error: " + test_error);
            System.out.println("Correct: " + correct);
            System.out.println("Incorrect: " + incorrect);

            System.out.printf("Percentage of correctly classifed samples: %.3f\n", accuracy);
        }

        return accuracy;
    }

    static double delta_sigmoid(double x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    static double sigmoid(double x)
    {
        return 1 / (1 + Math.exp(-x));
    }

    static void showHelp()
    {
        System.out.println("[USAGE]: java Regression TRAIN_DATA TEST_DATA VALIDATION_DATA [options]");
        System.out.println("Options:");
        System.out.println("-v --verbose: Verbose output");
        System.out.println("-h --help: Show this help message");
    }

    public static void main(String[] args)
    {
        Utility uObj = new Utility(); // Utility object
        boolean verbose = false;

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
                    verbose = true;
                if (options.equals("-h") || options.equals("--help"))
                {
                    showHelp();
                    System.exit(0);
                }
            }catch(Exception e){}
        } catch(Exception e) {
            showHelp();
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

        int[] avg_iterations = new int[6];
        double[] avg_accuracy = new double[6];

        for (int trials = 0; trials < 10; trials++) // Take average over 10 trials
        {
            System.out.println("Trial " + (trials+1));
            for (int hidden = 5; hidden <= 10; hidden++)
            {
                ANN ann = new ANN(hidden, verbose);
                int num_iterations = ann.train(x, 100, 3000, validation_x);
                double accuracy = ann.evaluateModel(test_x);
                System.out.printf("%d hidden neurons, %d iterations, %.3f accuracy\n", hidden, num_iterations, accuracy);
                avg_iterations[hidden-5] += num_iterations;
                avg_accuracy[hidden-5] += accuracy;
            }
        }
        System.out.println("----------------------------\nAverage statistics:");
        for (int hidden = 5; hidden <= 10; hidden++)
        {
            avg_iterations[hidden-5] /= 10;
            avg_accuracy[hidden-5] /= 10;
            System.out.printf("%d hidden neurons, %d iterations, %.3f accuracy\n", hidden, avg_iterations[hidden-5], avg_accuracy[hidden-5]);
        }



    }
}
