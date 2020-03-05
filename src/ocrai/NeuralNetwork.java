package ocrai;

import java.util.Objects;
import java.util.Random;

class NeuralNetwork {

    // Stores How many neurons each layer has
    private final int[] NEURON_COUNT;
    // The amount of inputs that go into the first layer
    private final int INPUT_SIZE;
    // The amount of outputs outputted from the last layer
    private final int OUTPUT_SIZE;
    // Stores the amount of layers in the NN
    private final int NUMBER_OF_LAYERS;
    // Training Loops to run
    private final static int TRAINING_LOOPS = 150;
    // the learning rate/ETA/Alpha/Epsilon
    private final static double ETA = 0.7;
    // Upper limit of randomly assigned weights
    private final static double WEIGHT_MAX = 1.5;
    // Lower limit of randomly assigned weights
    private final static double WEIGHT_LOWEST = -1.5;
    // Upper limit of randomly assigned bias
    private final static double BIAS_MAX = 1.5;
    // Lower limit of randomly assigned bias
    private final static double BIAS_LOWEST = -1.5;
    // First one stores the layer, second is the neuron I'm working with. Every neuron has only one output
    //= output[layer][neuron in layer 1]
    private double[][] output;
    // Every neuron has only one bias, so the same structure as for output is followed. First one is layer,
    // second is the neuron
    //= bias[layer][neuron in layer 1]
    private double[][] bias;
    //= weights[layer][neuronX][previousNeuron that neuronX is connected to in previous layer]
    // In the case of the BackPropagationError the sequence is [layer+1][previousNeuron][currentNeuron]
    private double[][][] weights;
    private double[][] errorChange;
    // The gradient with respect to that particular weight
    private double[][] partialDerivative;
    // The output layer number;
    private static int outputLayer;

    private static double averageAccuracy;

    private NeuralNetwork(int... NEURON_COUNT) {
        this.NEURON_COUNT = NEURON_COUNT;
        NUMBER_OF_LAYERS = NEURON_COUNT.length;
        INPUT_SIZE = NEURON_COUNT[0];
        OUTPUT_SIZE = NEURON_COUNT[NEURON_COUNT.length - 1];
        weights = new double[NUMBER_OF_LAYERS][][];
        bias = new double[NUMBER_OF_LAYERS][];
        outputLayer = NUMBER_OF_LAYERS-1;

        // Each neuron has one output, one error change and one output derivative
        output = new double[NUMBER_OF_LAYERS][];
        errorChange = new double[NUMBER_OF_LAYERS][];
        partialDerivative = new double[NUMBER_OF_LAYERS][];

        for (int thisLayer = 0; thisLayer < NUMBER_OF_LAYERS; thisLayer++) {
            // Determining how big each layer is. Each Neuron will have an output, error change and output_derivative
            // so they will be the same size
            output[thisLayer] = new double[NEURON_COUNT[thisLayer]];
            errorChange[thisLayer] = new double[NEURON_COUNT[thisLayer]];
            partialDerivative[thisLayer] = new double[NEURON_COUNT[thisLayer]];

            // Assigning random bias to each neuron
            bias[thisLayer] = new Random().doubles(NEURON_COUNT[thisLayer], BIAS_LOWEST, BIAS_MAX).toArray();
            // Weights array does not start at 0 as that is the input layer which has no weights.
            if (thisLayer > 0) {
                // Network_Layer_Sizes[i] refers to the current layer while the -1 refers to the previous layer.

                // Assigning random weights to each neuron
                weights[thisLayer] = NetworkOperations.assignRandomWeights(NEURON_COUNT[thisLayer], NEURON_COUNT[thisLayer - 1],
                        WEIGHT_LOWEST, WEIGHT_MAX);
            }
        }
    }

    public static void runNetworkAlgorithm(double[][] dataSet1, String dataSet1Name, double[][] dataSet2,
                                           String dataSet2Name) {
        // Create the network with any specified number of hidden layers. The number of neurons per layer is given
        NeuralNetwork neuralNetwork = new NeuralNetwork(64, 45, 10);
        // Train the network on dataset1 & test on dataset2
        launchTrainingAndTest(neuralNetwork, dataSet1, dataSet1Name, dataSet2, dataSet2Name);
        // Reverse the process & Train on the second file and test on the first file
        launchTrainingAndTest(neuralNetwork, dataSet2, dataSet2Name, dataSet1, dataSet1Name);
        System.out.println("\nAverage Accuracy is " + Main.decimalFormat.format(averageAccuracy/2) + "%");
    }

    private static void launchTrainingAndTest(NeuralNetwork neuralNetwork, double[][] trainFile, String trainFileLabel,
                                              double[][] testFile, String testFileLabel) {
        // Create a training set instance
        NeuralNetworkDataSet set = NetworkOperations.prepareDataForNeuralNetwork(trainFile, trainFileLabel);
        // Train the previously created training instance
        neuralNetwork.train(set);
        // Create a test set
        NeuralNetworkDataSet testSet = NetworkOperations.prepareDataForNeuralNetwork(testFile, testFileLabel);
        // test the training instance against the test instance
        testOnDataSet(neuralNetwork, testSet);
    }

    private double[] feedForwardMethod(double... input) {

        if (input.length != this.INPUT_SIZE)
            return null;
        // Setting first layer as the input layer. No calculations are carried out on this so it outputs the unchanged
        // input layer
        this.output[0] = input;
        // Start at the first HIDDEN layer
        for (int currentLayer = 1; currentLayer < NUMBER_OF_LAYERS; currentLayer++) {
            // Iterate for every neuron inside of that layer. Network_Layer_Sizes contains the amount of neuron,
            // thus need to iterate
            // For the amount of neurons that particular layer (specified above) contains
            for (int currentNeuron = 0; currentNeuron < NEURON_COUNT[currentLayer]; currentNeuron++) {
                // Adding the bias. This occurs once for every neuron in the current layer
                double sum = bias[currentLayer][currentNeuron];
                // Iterating through the neurons in the previous layer
                for (int previousNeuron = 0; previousNeuron < NEURON_COUNT[currentLayer - 1]; previousNeuron++) {
                    // Summing the output of the previous neuron while multiplying it with the weight that links the
                    // current neuron & previous neuron
                    sum += output[currentLayer - 1][previousNeuron]
                            * weights[currentLayer][currentNeuron][previousNeuron];
                }
                // Changing the output of the currently selected neuron through the sigmoid activation function
                output[currentLayer][currentNeuron] = 1d / (1 + Math.exp(-sum));
            }
        }
        // Returning the output of the network at the LAST layer.
        return output[outputLayer];
    }

    private void backPropagationError(double[] target) {
        // This is only for the output layer which is the last layer, which is why we do NetworkSize-1
        for (int currentNeuron = 0; currentNeuron < NEURON_COUNT[outputLayer]; currentNeuron++) {
            //The partial derivative of the logistic function is the output multiplied by 1 minus the output:
            partialDerivative[outputLayer][currentNeuron] = output[outputLayer][currentNeuron] *
                    (1 - output[outputLayer][currentNeuron]);
            // -(target - out) is expressed below as out-target. This is then multiplied by the partial derivative
            // obtained by multiplying the output by 1 minus the output.
            // out-target refers to how much the total error changes with regards the total output.
            errorChange[outputLayer][currentNeuron] = (output[outputLayer][currentNeuron] - target[currentNeuron])
                    // Multiply the error change with regards the total output  with the partial derivative of the
                    // current neuron to get the value that will be used along with the delta rule to update the current
                    // neuron's weights
                    * partialDerivative[outputLayer][currentNeuron];
        }
        // <<< Taking care of the Hidden Layers >>>
        // Starting from the last hidden layer, which is ONE layer before the output
        // layer (outputlayer - 1)
        // Layer 0 is the input layer so I'm not going to loop through the 0 layer (No processes have taken place
        // on that layer)
        for (int currentLayer = outputLayer - 1; currentLayer > 0; currentLayer--) {
            // Need to go through each neuron in this layer
            for (int currentNeuron = 0; currentNeuron < NEURON_COUNT[currentLayer]; currentNeuron++) {
                // Sum over the weights multiplied by their error changes. At first it is equal to 0
                double sum = 0;
                // Going through the neurons in the next layer that have already and their error changes calculated.
                // The first hidden layer will loop through the output layer
                for (int neuronInNextLayer = 0; neuronInNextLayer < NEURON_COUNT[currentLayer + 1]; neuronInNextLayer++) {
                    // Increase the sum between the weight that connects the current neuron to the next neuron
                    // Since this time the back propagation error is being calculated backwards I'm checking the weight of the
                    // next neuron to this neuron. It seems like it is going "backwards" but the "next" neuron in this
                    // case is backwards so the opposite of the feedForward method
                    sum += weights[currentLayer + 1][neuronInNextLayer][currentNeuron] *
                            errorChange[currentLayer + 1][neuronInNextLayer];
                }
                // Calculating the partial derivative with the same formula as that used in the output layer
                partialDerivative[currentLayer][currentNeuron] = output[currentLayer][currentNeuron] *
                        (1 - output[currentLayer][currentNeuron]);
                // As the sum is the error change of each individual neuron with regards to the previous neurons,
                // multiplying it with the partial derivative of the current neuron will give the sum of the error
                // changes. This is repeated for each neuron in this layer and added to this error change for the
                // selected neuron is the error change with regards all of the previous layer's neurons
                errorChange[currentLayer][currentNeuron] = sum * partialDerivative[currentLayer][currentNeuron];
            }
        }
    }

    private void updateWeights() {
        // Starting from the first hidden layer
        for (int currentLayer = 1; currentLayer < NUMBER_OF_LAYERS; currentLayer++) {
            for (int currentNeuron = 0; currentNeuron < NEURON_COUNT[currentLayer]; currentNeuron++) {

                // Iterating through every neuron in the previous layer to get each weight that connects the current
                // neuron to the previous neuron
                for (int previousLayerNeuron = 0; previousLayerNeuron < NEURON_COUNT[currentLayer - 1];
                     previousLayerNeuron++) {
                    // The weight I'm dealing with is the weight of the current layer's current neuron connected to the
                    // previous neuron.
                    // To decrease the error, The errorChange is subtracted from the current weight & multiplied by the
                    // learning weight (ETA)
                    double delta = (-ETA * errorChange[currentLayer][currentNeuron]) *
                            output[currentLayer - 1][previousLayerNeuron];
                    // Delta is applied to weights to update them;
                    weights[currentLayer][currentNeuron][previousLayerNeuron] += delta ;
                }
                // Can be outside of the loop as every neuron does not have a bias linked to the previous neuron.
                // Updating the bias helps reduce the chance of getting stuck in local optima
                bias[currentLayer][currentNeuron] += -ETA * errorChange[currentLayer][currentNeuron];
            }
        }
    }

    // The train method which calls the respective methods

    private void train(NeuralNetworkDataSet neuralNetworkDataSet) {
        if (neuralNetworkDataSet.INPUT_SIZE != INPUT_SIZE || neuralNetworkDataSet.OUTPUT_SIZE != OUTPUT_SIZE) return;
        for (int i = 0; i < TRAINING_LOOPS; i++) {
            for (int inputSize = 0; inputSize < neuralNetworkDataSet.getSize(); inputSize++) {
                feedForwardMethod(neuralNetworkDataSet.getInput(inputSize));
                backPropagationError(neuralNetworkDataSet.getOutput(inputSize));
                updateWeights();
            }
        }
    }



    private static void testOnDataSet(NeuralNetwork neuralNetwork, NeuralNetworkDataSet set) {
        int correct = 0;
        for (int i = 0; i < set.getSize(); i++) {
            // Checking whether the output from the feed forward method matches the expected output
            // (The label assigned to that row)
            if (NetworkOperations.indexOfMax(Objects.requireNonNull(neuralNetwork.feedForwardMethod(set.getInput(i))))
                    == NetworkOperations.indexOfMax((set.getOutput(i))))
                correct++;
        }

        System.out.printf("\n<< %s >> Dataset tested\n<< %d >> Correct answers\n<< %d >> Incorrect answers \n" +
                        "<< %s%% >> Total Accuracy %n", set.LABEL, correct, set.getSize() - correct,
                Main.decimalFormat.format(((double) correct / (double) set.getSize()) * 100));
        averageAccuracy+= ((double) correct / (double) set.getSize()) * 100;
    }
}